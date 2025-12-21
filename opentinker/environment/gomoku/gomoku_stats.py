#!/usr/bin/env python3
"""Gomoku Game Statistics Tracker.

This module provides Gomoku-specific game statistics tracking including:
- Win rate (player X wins)
- Loss rate (player O wins)  
- Draw rate
- Timeout rate

Usage:
    from gomoku_stats import GomokuGameStats
    
    stats = GomokuGameStats()
    stats.record_game_result(info, reward, done)
    step_stats = stats.get_step_stats()
"""

from typing import Any, Dict, List

from opentinker.environment.base_game_server import BaseGameStats


class GomokuGameStats(BaseGameStats):
    """Statistics tracker for Gomoku game server.
    
    Extends BaseGameStats with Gomoku-specific metrics:
    - win_rate: Percentage of games won by X (the LLM)
    - loss_rate: Percentage of games won by O (the opponent)
    - draw_rate: Percentage of games ending in draw
    - timeout_rate: Percentage of games ending in timeout
    """
    
    def record_game_result(self, info: Dict[str, Any], reward: float, done: bool, instance_id: str = "default", job_id: str = "default"):
        """Record a Gomoku game result.
        
        Args:
            info: Game info dict with winner/draw/error fields
            reward: Reward for this step
            done: Whether the game is finished
            instance_id: Game instance identifier for per-game tracking
            job_id: Job identifier for logging (for compatibility with base class)
        """
        with self._lock:
            # Track that this game has started
            self._started_games.add(instance_id)
            
            # Initialize game reward tracking if needed
            if instance_id not in self._game_rewards:
                self._game_rewards[instance_id] = {"sum_reward": 0.0, "step_count": 0}
            
            # Accumulate reward for this game
            self._game_rewards[instance_id]["sum_reward"] += reward
            self._game_rewards[instance_id]["step_count"] += 1
            
            if done:
                # Calculate per-game statistics
                game_data = self._game_rewards[instance_id]
                sum_reward = game_data["sum_reward"]
                step_count = game_data["step_count"]
                avg_reward = sum_reward / step_count if step_count > 0 else 0.0
                
                # Record completed game stats
                self._step_stats["game_completed"].append(1.0)
                self._step_stats["final_rewards"].append(reward)
                self._step_stats["sum_rewards"].append(sum_reward)
                self._step_stats["avg_rewards"].append(avg_reward)
                self._step_stats["game_step_counts"].append(float(step_count))
                
                winner = info.get("winner")
                if winner == "X":
                    self._step_stats["wins"].append(1.0)
                    self._step_stats["losses"].append(0.0)
                    self._step_stats["draws"].append(0.0)
                    self._step_stats["timeouts"].append(0.0)
                    self._cumulative_stats["total_wins"] += 1
                elif winner == "O":
                    self._step_stats["wins"].append(0.0)
                    self._step_stats["losses"].append(1.0)
                    self._step_stats["draws"].append(0.0)
                    self._step_stats["timeouts"].append(0.0)
                    self._cumulative_stats["total_losses"] += 1
                elif info.get("draw"):
                    self._step_stats["wins"].append(0.0)
                    self._step_stats["losses"].append(0.0)
                    self._step_stats["draws"].append(1.0)
                    self._step_stats["timeouts"].append(0.0)
                    self._cumulative_stats["total_draws"] += 1
                elif info.get("error") == "timeout":
                    self._step_stats["wins"].append(0.0)
                    self._step_stats["losses"].append(0.0)
                    self._step_stats["draws"].append(0.0)
                    self._step_stats["timeouts"].append(1.0)
                    self._cumulative_stats["total_timeouts"] += 1
                else:
                    # Unknown outcome - treat as timeout/error
                    self._step_stats["timeouts"].append(1.0)
                    self._cumulative_stats["total_timeouts"] += 1
                
                self._cumulative_stats["total_games"] += 1
                
                # Clear this game's tracking (game is done)
                del self._game_rewards[instance_id]
            
            # Track ALL step rewards
            self._step_stats["all_step_rewards"].append(reward)
    
    def get_step_stats(self) -> Dict[str, Any]:
        """Get Gomoku-specific statistics for current step."""
        with self._lock:
            stats = {"step": self._current_step}
            
            # Get accumulated values
            wins = self._step_stats.get("wins", [])
            losses = self._step_stats.get("losses", [])
            draws = self._step_stats.get("draws", [])
            timeouts = self._step_stats.get("timeouts", [])
            completed = self._step_stats.get("game_completed", [])
            final_rewards = self._step_stats.get("final_rewards", [])
            sum_rewards = self._step_stats.get("sum_rewards", [])
            avg_rewards = self._step_stats.get("avg_rewards", [])
            game_step_counts = self._step_stats.get("game_step_counts", [])
            all_rewards = self._step_stats.get("all_step_rewards", [])
            
            # Total samples = all games that started
            total_samples = len(self._started_games)
            # Incomplete = games that started but haven't completed yet
            incomplete_samples = len(self._game_rewards)
            # Completed games
            completed_count = len(completed)
            
            stats["total_samples"] = total_samples
            stats["games_in_step"] = completed_count
            stats["incomplete_samples"] = incomplete_samples
            stats["completion_rate"] = completed_count / total_samples if total_samples > 0 else 0.0
            
            if completed_count > 0:
                stats["win_rate"] = sum(wins) / completed_count
                stats["loss_rate"] = sum(losses) / completed_count
                stats["draw_rate"] = sum(draws) / completed_count
                stats["timeout_rate"] = sum(timeouts) / completed_count
            else:
                stats["win_rate"] = 0.0
                stats["loss_rate"] = 0.0
                stats["draw_rate"] = 0.0
                stats["timeout_rate"] = 0.0
            
            # Mean of FINAL rewards only (last reward of each game)
            if final_rewards:
                stats["mean_final_reward"] = sum(final_rewards) / len(final_rewards)
            else:
                stats["mean_final_reward"] = 0.0
            
            # Mean of SUM rewards (total reward per game) - COMPLETED games only
            if sum_rewards:
                stats["mean_sum_reward"] = sum(sum_rewards) / len(sum_rewards)
            else:
                stats["mean_sum_reward"] = 0.0
            
            # Mean of SUM rewards for ALL games (including incomplete ones)
            # This should match val/mean_score from the reward function
            all_sum_rewards = list(sum_rewards)  # Start with completed games
            for game_data in self._game_rewards.values():
                all_sum_rewards.append(game_data["sum_reward"])
            
            if all_sum_rewards:
                stats["mean_sum_reward_all"] = sum(all_sum_rewards) / len(all_sum_rewards)
            else:
                stats["mean_sum_reward_all"] = 0.0
            
            # Mean of AVG rewards (average reward per step in each game)
            if avg_rewards:
                stats["mean_avg_reward"] = sum(avg_rewards) / len(avg_rewards)
            else:
                stats["mean_avg_reward"] = 0.0
            
            # Mean game length
            if game_step_counts:
                stats["mean_game_steps"] = sum(game_step_counts) / len(game_step_counts)
            else:
                stats["mean_game_steps"] = 0.0
            
            # Keep backward compatibility
            stats["mean_reward"] = stats["mean_final_reward"]
            stats["total_interactions"] = len(all_rewards)
            
            return stats
    
    def get_cumulative_stats(self) -> Dict[str, Any]:
        """Get Gomoku-specific cumulative statistics."""
        with self._lock:
            stats = dict(self._cumulative_stats)
            total = stats.get("total_games", 0)
            if total > 0:
                stats["cumulative_win_rate"] = stats.get("total_wins", 0) / total
                stats["cumulative_loss_rate"] = stats.get("total_losses", 0) / total
                stats["cumulative_draw_rate"] = stats.get("total_draws", 0) / total
            return stats

