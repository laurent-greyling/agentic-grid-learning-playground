# Agentic Grid Learning Playground

This repository is a learning project focused on understanding agentic behaviour, online learning and decision-making with behavioural patterns without relying on heavy machine-learning frameworks.

The goal was not to build “the best model”, but to build something I could fully explain, reason about and iterate on, step by step.

## 1. Why build this

I wanted to understand:
 - How an agent can form beliefs about an environment
 - How learning happens when feedback arrives after each action
 - How exploration vs exploitation affects long-term behaviour
 - Why some statistical choices (mean vs median) behave better in noisy systems

Rather than starting with neural networks or black-box reinforcement learning libraries, I deliberately built everything from first principles:
 - explicit state
 - explicit decisions
 - explicit learning updates

This makes the behaviour observable, debuggable and explainable.

## 2. What does the app do

The environment is a 100 × 100 grid.

Each round:
 1. A flag appears somewhere on the grid
 2. The agent guesses the flag location
 3. The true location is revealed
 4. The agent updates its internal belief
 5. Performance is measured using Manhattan distance

### Environment bias (hotspot)

Most flags are generated around a hotspot:

```python
HOTSPOT_CENTER_X_IN_BLOCKS = 70
HOTSPOT_CENTER_Y_IN_BLOCKS = 25
HOTSPOT_STANDARD_DEVIATION_IN_BLOCKS = 8.0
HOTSPOT_PROBABILITY = 0.8
```

This hotspot is:
 - not known to the agent
 - not symmetric
 - intentionally off-center

The asymmetry makes learning visible and prevents accidental “lucky guesses” caused by symmetry.

## 3. Agent Behaviour (Exploration vs Exploitatation)

The agent uses an epsilon-greedy policy:
 - With probability ε → explore
 - Otherwise → exploit current belief

Early on:
 - ε is high → more exploration

Over time:
 - ε decays → more exploitation

Exploration is local, not random across the whole grid.

This mirrors how a human would behave: “If I believe the flag is around here, I’ll search nearby not on the other side of the map.”

## 4. Why does online learning matter

The agent learns online:
 - one observation at a time
 - immediately after each round

There is:
 - no training phase
 - no reset between runs
 - no batch reprocessing

The model state is stored in `model_state.json` and continues learning across executions.

This was a deliberate choice:
 - restarting learning every run hides long-term behaviour
 - online learning exposes convergence, drift and stability

 ## 5. Why median instead of mean

 Early versions of this project used the mean of observed positions.

That worked, but it was unstable.

The problem with mean
 - Large outliers (rare far-away flags) pull the belief strongly
 - The belief “wobbles” even after long learning
 - Performance degrades in noisy environments

### Why median works better here

The belief is now based on the median X and Y positions computed from a 2D observation histogram.

Median:
 - handles outliers better
 - converges faster
 - stabilises belief in the densest region

This matches the environment:
 - most observations cluster near the hotspot
 - occasional far flags should not dominate belief

The result is:
 - smoother convergence
 - more human-like behaviour
 - lower long-term average distance

 ## 6. Why the Average levels out

Even with a good learner, the average Manhattan distance never goes to zero.

This is expected.

Reasons:
 - The environment is probabilistic
 - Flags sometimes appear far from the hotspot
 - Exploration never fully stops
 - Distance measures error magnitude, not correctness

A good learner doesn’t eliminate error, it minimises expected error.

The plateau you see is the best achievable performance given:
 - the environment’s randomness
 - the chosen policy
 - the grid size

## 7. Visualising learning

After each run, the app can generate visualisations showing:
 - Belief vs true X over time
 - Belief vs true Y over time
 - Distance per round
 - Session average distance (learning curve)

This makes learning behaviour visible, not just numeric.

## 8. Running the app

### Requirements
 - Python 3,10+
 - `matplotlib`

 ### Setup

 ```bash
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib
 ```

 ### Run

 ```bash
 python main.py
 ```

 To reset learning

 ```bash
 rm model_state.json
 ```

 ## 9. What this project is (and isn't)

This project is:
- a learning exercise
- an agentic system
- explainable and inspectable
- intentionally simple

This project is not:
 - a production ML model
 - a neural network
 - an optimisation benchmark

The value is in understanding, not raw performance.

## Conclusion

This project evolved through multiple iterations:
- naive guessing
- mean-based belief
- online learning
- local exploration
- median-based belief

Each change was driven by observing behaviour and asking:

`Does this make sense if a human were doing it?`

That question guided every design decision.
