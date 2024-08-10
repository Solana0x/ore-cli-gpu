use std::{sync::Arc, time::Instant};
use std::sync::atomic::{AtomicU32, Ordering};

use colored::*;
use drillx::{Hash, Solution};
use ore_api::{
    consts::{BUS_ADDRESSES, BUS_COUNT, EPOCH_DURATION},
    state::{Config, Proof},
};
use rand::Rng;
use solana_program::pubkey::Pubkey;
use solana_rpc_client::spinner;
use solana_sdk::signer::Signer;

use crate::{
    args::MineArgs,
    send_and_confirm::ComputeBudget,
    utils::{amount_u64_to_string, get_clock, get_config, get_proof_with_authority, proof_pubkey},
    Miner,
};

#[cfg(feature = "gpu")]
extern "C" {
    pub static BATCH_SIZE: u32;
    pub fn hash(challenge: *const u8, nonce: *const u8, out: *mut u64);
    pub fn solve_all_stages(hashes: *const u64, out: *mut u8, sols: *mut u32, num_sets: i32);
}

impl Miner {
    pub async fn mine(&self, args: MineArgs) {
        let signer = self.signer();
        self.open().await;
        self.check_num_cores(args.threads);

        loop {
            let proof = get_proof_with_authority(&self.rpc_client, signer.pubkey()).await;
            println!(
                "\nStake balance: {} ORE",
                amount_u64_to_string(proof.balance)
            );

            let cutoff_time = self.get_cutoff(proof, args.buffer_time).await;
            let config = get_config(&self.rpc_client).await;
            let solution = Self::find_hash_par(
                proof,
                cutoff_time,
                args.threads,
                config.min_difficulty as u32,
            )
            .await;

            let mut compute_budget = 500_000;
            let mut ixs = vec![ore_api::instruction::auth(proof_pubkey(signer.pubkey()))];
            if self.should_reset(config).await {
                compute_budget += 100_000;
                ixs.push(ore_api::instruction::reset(signer.pubkey()));
            }
            ixs.push(ore_api::instruction::mine(
                signer.pubkey(),
                signer.pubkey(),
                find_bus(),
                solution,
            ));
            self.send_and_confirm(&ixs, ComputeBudget::Fixed(compute_budget), false)
                .await
                .ok();
        }
    }

    #[cfg(not(feature = "gpu"))]
    async fn find_hash_par(
        proof: Proof,
        cutoff_time: u64,
        threads: u64,
        min_difficulty: u32,
    ) -> Solution {
        let progress_bar = Arc::new(spinner::new_progress_bar());
        progress_bar.set_message("Mining...");

        let handles: Vec<_> = (0..threads)
            .map(|i| {
                std::thread::spawn({
                    let proof = proof.clone();
                    let progress_bar = progress_bar.clone();
                    let mut memory = equix::SolverMemory::new();
                    move || {
                        let timer = Instant::now();
                        let mut nonce = u64::MAX.saturating_div(threads) * i;
                        let mut best_nonce = nonce;
                        let mut best_difficulty = 0;
                        let mut best_hash = Hash::default();
                        loop {
                            if let Ok(hx) = drillx::hash_with_memory(
                                &mut memory,
                                &proof.challenge,
                                &nonce.to_le_bytes(),
                            ) {
                                let difficulty = hx.difficulty();
                                if difficulty.gt(&best_difficulty) {
                                    best_nonce = nonce;
                                    best_difficulty = difficulty;
                                    best_hash = hx;
                                }
                            }

                            if nonce % 100 == 0 {
                                if timer.elapsed().as_secs() >= cutoff_time {
                                    if best_difficulty > min_difficulty {
                                        break;
                                    }
                                } else if i == 0 {
                                    progress_bar.set_message(format!(
                                        "Mining... ({} sec remaining)",
                                        cutoff_time.saturating_sub(timer.elapsed().as_secs()),
                                    ));
                                }
                            }
                            nonce += 1;
                        }
                        (best_nonce, best_difficulty, best_hash)
                    }
                })
            })
            .collect();

        let mut best_nonce = 0;
        let mut best_difficulty = 0;
        let mut best_hash = Hash::default();
        for h in handles {
            if let Ok((nonce, difficulty, hash)) = h.join() {
                if difficulty > best_difficulty {
                    best_difficulty = difficulty;
                    best_nonce = nonce;
                    best_hash = hash;
                }
            }
        }

        progress_bar.finish_with_message(format!(
            "Best hash: {} (difficulty: {})",
            bs58::encode(best_hash.h).into_string(),
            best_difficulty
        ));

        Solution::new(best_hash.d, best_nonce.to_le_bytes())
    }

    #[cfg(feature = "gpu")]
    async fn find_hash_par(
        proof: Proof,
        cutoff_time: u64,
        threads: u64,
        min_difficulty: u32,
    ) -> Solution {
        let progress_bar = Arc::new(spinner::new_progress_bar());
        progress_bar.set_message("Mining with GPU...");

        let timer = Instant::now();
        let proof = proof.clone();

        const INDEX_SPACE: usize = 65536;
        let x_batch_size = unsafe { BATCH_SIZE as usize };

        let mut hashes = vec![0u64; x_batch_size * INDEX_SPACE];
        let mut x_nonce = 0u64;
        let mut processed = 0;

        let xbest = Arc::new(AtomicU32::new(0));

        loop {
            unsafe {
                hash(
                    proof.challenge.as_ptr(),
                    &x_nonce as *const u64 as *const u8,
                    hashes.as_mut_ptr() as *mut u64,
                );
            }

            let mut digest = vec![0u8; x_batch_size * 16];
            let mut sols = vec![0u32; x_batch_size];

            unsafe {
                solve_all_stages(
                    hashes.as_ptr(),
                    digest.as_mut_ptr(),
                    sols.as_mut_ptr(),
                    x_batch_size as i32,
                );
            }

            let chunk_size = x_batch_size / threads as usize;
            (0..threads as usize).into_par_iter().for_each(|i| {
                let start = i * chunk_size;
                let end = if i + 1 == threads as usize { x_batch_size } else { start + chunk_size };

                for i in start..end {
                    if sols[i] > 0 {
                        let solution_digest = &digest[i * 16..(i + 1) * 16];
                        let solution = Solution::new(solution_digest.try_into().unwrap(), (x_nonce + i as u64).to_le_bytes());
                        let difficulty = solution.to_hash().difficulty();
                        if solution.is_valid(&proof.challenge) && difficulty > xbest.load(Ordering::Relaxed) {
                            xbest.store(difficulty, Ordering::Relaxed);
                        }
                    }
                }
            });

            x_nonce += x_batch_size as u64;
            processed += x_batch_size;

            let elapsed = timer.elapsed().as_secs();
            let best_difficulty = xbest.load(Ordering::Relaxed);

            progress_bar.set_message(format!(
                "Mining with GPU... (Best difficulty: {}, Time Remaining: {}s, Processed: {})",
                best_difficulty,
                cutoff_time.saturating_sub(elapsed),
                processed
            ));

            if timer.elapsed().as_secs() >= cutoff_time {
                if best_difficulty > min_difficulty {
                    break;
                }
            }
        }

        let final_best = xbest.load(Ordering::Relaxed);

        // Construct the digest by combining final_best with part of the hash and padding
        let mut final_digest = [0u8; 16];

        // Copy the bytes of `final_best` into the first 4 bytes of `final_digest`
        final_digest[0..4].copy_from_slice(&final_best.to_le_bytes());

        // Optionally, include part of the hash result in the digest
        if hashes.len() > 0 {
            final_digest[4..12].copy_from_slice(&hashes[0].to_le_bytes());
        }

        progress_bar.finish_with_message(format!(
            "Best hash: {} (difficulty: {})",
            bs58::encode(final_digest).into_string(),
            final_best
        ));

        Solution::new(final_digest, x_nonce.to_le_bytes())
    }

    pub fn check_num_cores(&self, threads: u64) {
        let num_cores = num_cpus::get() as u64;
        if threads > num_cores {
            println!(
                "{} Number of threads ({}) exceeds available cores ({})",
                "WARNING".bold().yellow(),
                threads,
                num_cores
            );
        }
    }

    async fn should_reset(&self, config: Config) -> bool {
        let clock = get_clock(&self.rpc_client).await;
        config
            .last_reset_at
            .saturating_add(EPOCH_DURATION)
            .saturating_sub(5)
            .le(&clock.unix_timestamp)
    }

    async fn get_cutoff(&self, proof: Proof, buffer_time: u64) -> u64 {
        let clock = get_clock(&self.rpc_client).await;
        proof
            .last_hash_at
            .saturating_add(60)
            .saturating_sub(buffer_time as i64)
            .saturating_sub(clock.unix_timestamp)
            .max(0) as u64
    }
}

fn find_bus() -> Pubkey {
    let i = rand::thread_rng().gen_range(0..BUS_COUNT);
    BUS_ADDRESSES[i]
}
