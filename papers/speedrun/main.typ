#import "../common/template.typ": paper, note
#import "generated/keys.typ": results

#show: paper.with(
  title: [Learning to Speedrun a Long-Horizon Farming Objective from Trajectories],
  authors: ((name: "dotrdp", affil: "Independent"), (name: "Claude (Anthropic)", affil: "AI coding assistant")),
  abstract: [
    We study how well a policy can learn to *speedrun* a long-horizon collection objective — completing all 24 restoration requests of the Furrow farming simulation as early as possible — primarily from demonstrations. The objective couples seasonal crop windows, weather- and time-gated resources, tool progression and money, and a single missed window costs a season. Using a stateless scripted expert as demonstrator and interactive oracle, we compare behaviour cloning with data scaling, noise-injected demonstrations (DART), filtered and time-conditioned imitation, DAgger, and PPO finetuning with a KL penalty to the imitation prior, all evaluated on #results.n_test paired held-out seeds by the censored completion day. The scripted expert completes the objective on #calc.round(results.expert_success * 100, digits: 1)% of test seeds in #calc.round(results.expert_mean_C, digits: 1) days on average, and a masked random policy never does. The best learned policy, #results.best_name, succeeds on #calc.round(results.best_success * 100, digits: 1)% of seeds with a mean censored completion day of #calc.round(results.best_mean_C, digits: 1). We describe the reward and evaluation choices that make these comparisons meaningful, and a SMAPI bridge and NeMo Gym server that extend the setup to the commercial game the benchmark is modelled on and to LLM agents.
  ],
  keywords: ("imitation learning", "DAgger", "speedrunning", "long-horizon RL", "behaviour cloning"),
)

= Introduction

Speedrunning turns a game into an optimisation problem with a single scalar objective: finish as early as possible. Farming/life simulations make this objective unusually long-horizon. Crops planted too late in a season die, some fish bite only in the rain or at night, a wandering merchant's random stock can shortcut an item that would otherwise require waiting for the next season, and an early tool upgrade can unlock deeper ores weeks later. Human speedrunners of the restoration ("community center") objective in the commercial game Stardew Valley plan routes around exactly these constraints.

We ask how much of such a route can be *learned from trajectories*. Following the VPT recipe of imitation followed by RL finetuning @baker2022vpt and lessons from NetHack, where behaviour cloning from billions of transitions recovers only part of the demonstrator's ability @hambro2022dungeons @piterbarg2023hihack, we work in Furrow — an original, deterministic, fast simulator with the same structure (companion paper) — where demonstrations are cheap and the demonstrator can be queried interactively. This lets us separate the effects of data quantity, data diversity, covariate shift and RL finetuning.

= Problem

An episode starts on day 1 of spring with 500 coins. The agent acts through 192 *macro actions* (temporally extended options @sutton1999options: `Plant(crop, field)`, `Fish(spot)`, `Buy(item, q)`, `Deliver`, `Sleep`, …), each executed by a deterministic controller and masked when its preconditions fail. Observations are 777 item-centric features. The objective is the *completion day* $C$ on which all 24 requests are fulfilled; failures are censored at the 192-day cap. The reward is $-1$ per in-game day plus potential-based shaping on progress @ng1999shaping, and discounting is per in-game day ($gamma_"day" = 0.99$ raised to the number of day boundaries crossed by a transition), so the return-optimal policy is the fastest one and no policy profits from padding a day with extra decisions.

= Methods

*Demonstrator.* A stateless priority-rule expert (companion paper) that, for every open request, targets the cheapest candidate slots under a seasonal cost model and executes the first applicable rule. Because it depends only on the state, it provides clean labels on any state, including those visited by a learner.

*Data.* Three datasets on disjoint training seeds: 1,000 clean expert episodes (1.28 M decisions), 400 episodes with 10% injected random actions but clean labels (DART @laskey2017dart), and 400 episodes from seven perturbed expert *styles*.

*Policy.* A 3×512 MLP with layer normalisation over standardised observations (statistics frozen from the data) and masked logits; a separate critic trunk for RL.

*Behaviour cloning (BC).* Masked cross-entropy on clean labels with label smoothing restricted to valid actions, episode-balanced sampling, 20 k steps of batch 4096. We vary the number of expert episodes (25 → 1000), add DART data, add style data, *filter* to the fastest 25% of episodes, and *time-condition* on days-to-go @chen2021dt (successful episodes only; at test time we request a target completion day and decrement it as days pass).

*DAgger* @ross2011dagger. Starting from BC on 100 episodes, each round rolls out a $beta$-mixture of expert and learner ($beta_i = 0.5^(i+1)$) in 64 parallel environments for 1,500 decisions, labels all visited states with the expert, aggregates, and retrains; six rounds.

*PPO finetuning* @schulman2017ppo. From BC+DART: critic warm-up on rollouts of the frozen prior, then PPO with masked log-probabilities, GAE @schulman2016gae with per-day discounting and bootstrapping from terminal observations on truncation, and a KL penalty to the prior annealed from 0.2 to 0.02 (in the spirit of kickstarting @schmitt2018kickstarting).

*Evaluation.* #results.n_test held-out test seeds, identical for every method; because exogenous randomness is keyed by seed and day, every method faces the same weather, forage and merchant stock, so per-seed differences to the expert are paired. We report success rate, mean, median and interquartile mean (IQM) of $C$ with bootstrap confidence intervals @agarwal2021rliable, the speed score $S = (192 - overline(C)) / (192 - overline(C)_"expert")$ and the paired difference to the expert. Learned policies act greedily.

= Results

#figure(include "generated/main_table.typ", caption: [Test-seed results (one episode per seed, greedy decoding). $C$ is the censored completion day (lower is better); Δ is the paired per-seed difference to the scripted expert.]) <tab:main>

#figure(image("generated/completion_cdf.png", width: 78%), caption: [Fraction of test seeds on which the objective is complete by a given day. Curves that saturate below 1 failed on some seeds within 192 days.]) <fig:cdf>

#figure(image("generated/bc_scaling.png", width: 62%), caption: [Behaviour cloning versus the number of expert episodes. The dotted line is the expert's mean completion day.]) <fig:scaling>

RESULTS_DISCUSSION

= Toward the commercial game and LLM agents

*SMAPI bridge.* The same agent interface is designed to drive the user's own copy of Stardew Valley through a SMAPI @smapi mod — lockstep stepping, structured snapshots including Community Center bundle state, primitive input frames and macro actions executed with the game's own systems — without redistributing any game code or data. It is used to record human trajectories and to evaluate policies; Furrow and the commercial game share the interface shape but not the observation or action semantics, so we make no transfer claim.

*LLM agents.* An NVIDIA NeMo Gym @nemogym resources server exposes Furrow as tools (`observe`, `list_actions`, `act`, `status`) over *segments* of the speedrun (a start state taken from an expert trajectory and a seven-day horizon), with a lexicographic reward: any success outranks any failure, and faster successes score higher relative to the expert's time from the same state. Segments keep prompts within a few thousand tokens per decision, which full 1,000-decision episodes would not.

= Limitations

The demonstrator is a scripted planner, not human; its weaknesses (it sometimes ends days early while waiting for a seasonal window) bound what pure imitation can learn and leave room for RL. Policies are memoryless MLPs; sequence models, entity-centric action heads and offline RL (IQL @kostrikov2022iql) are natural next steps. Results are from a single training seed per method; the paired, seeded evaluation keeps comparisons meaningful but does not measure training variance.

#bibliography("../common/refs.bib", style: "ieee")
