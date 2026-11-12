# YouTube Script: Teaching a Neural Network to Solve a Rubik's Cube (in Your Browser)

**Working title:** I Taught an AI to Solve a Rubik's Cube Without Telling It How
**Alternative titles:**
- This AI Learned to Solve a Rubik's Cube by Scrambling It Backwards
- A Rubik's Cube Solver That Never Saw a Human Solve
- Training a Rubik's Cube AI in 15 Minutes (and Running It in the Browser)

**Target length:** about 10 minutes
**Format:** screen recording plus voiceover, with short code close-ups and simple animations

> **How to read this script:** **VISUAL** is what's on screen. **VO** is the narration. Text in `[brackets]` is a placeholder to fill in with your own numbers or footage before recording.

---

## 0. Cold open (0:00 – 0:30)

**VISUAL:** The browser page. Press **Scramble** and the cube twists into a mess. Press **Solve with AI**. The status reads "Thinking… depth 12, 13, 14…", then the cube solves itself move by move. It ends on "AI solved it in `[N]` moves!"

**VO:**
> This Rubik's cube just solved itself, and nobody taught it how.
> No beginner's method, no algorithms copied from the internet, and no existing solver running behind the scenes.
> It's a small neural network that learned by watching a cube get scrambled and figuring out how to run that backwards.
> It all runs in your browser, with no server.
> Here's how it works.

---

## 1. The problem (0:30 – 1:30)

**VISUAL:** Slow 3D orbit around a scrambled cube. Big on-screen text: **43,252,003,274,489,856,000 positions.**

**VO:**
> A Rubik's cube has about 43 quintillion possible positions. Only one of them is solved.
> If you checked a billion positions every second, going through all of them would take more than a thousand years.
> So brute force is out.

**VISUAL:** Split screen. On the left, a person following a beginner's method. On the right, a scrolling list of hand-written rules.

**VO:**
> The classic approach is to write rules by hand: solve the white cross, then the corners, then the middle layer, and so on.
> That works, but a human designed every step.
> I wanted something different: a program that learns to solve the cube with no human knowledge at all.

---

## 2. The big idea: scramble it backwards (1:30 – 3:00)

**VISUAL:** Animation of a solved cube taking one move, `R`. A label appears: *"To undo this: R'"*. Then another move, `U`, labelled *"To undo this: U'"*. Then several more.

**VO:**
> Here's the trick, and it comes from a 2023 paper called *EfficientCube*.
> Start with a solved cube and make a random move, say R.
> We already know how to get back from this position: do R-prime.
> Make another random move, U. From here, the move that heads back is U-prime.
> Keep going, and every position we pass through comes with a free answer key: the move that undoes the last turn.

**VISUAL:** Code close-up of [training/data.py](training/data.py), highlighting `generate_batch` and the line `labels[d] = INVERSE[moves]`.

**VO:**
> In code, it's only a few lines. We scramble 26 moves deep, and every position along the way becomes a training example, labelled with the inverse of the move that created it.
> Why 26? Any position can be solved in 26 quarter-turns or fewer, so scrambles this long reach every depth.
> One small detail: a scramble never immediately undoes its own last move, because R followed by R-prime is just wasted motion.

**VISUAL:** On-screen callout: *"Is the label always the BEST move? No, but on average it points home."*

**VO:**
> Is that label always the best move? No. A random scramble sometimes wanders in a circle, and a shorter way back can exist.
> But averaged over millions of examples, the label points toward solved. That's all the network needs.
> And because we create the data ourselves, there's no limit to it. Every training step makes a brand-new batch, so the network never sees the same data twice.

---

## 3. Teaching the computer what a cube is (3:00 – 4:00)

**VISUAL:** An exploded view of the cube with all 54 stickers numbered, then the stickers flattening into a row of 54 colored squares.

**VO:**
> Before we can train anything, the computer needs a way to represent the cube.
> I wrote a small cube simulator in Python with NumPy. A cube state is a list of 54 numbers, one for each sticker, and each number is a color.

**VISUAL:** Code close-up of [training/cube_env.py](training/cube_env.py), showing `_move_permutation` and `apply_moves`.

**VO:**
> Each of the 12 moves (U, D, L, R, F, B and their primes) becomes a *permutation*: a lookup table that says which sticker moves where.
> So applying a move is one array lookup, and NumPy can turn thousands of cubes at once. That's what makes generating data so fast.

**VISUAL:** Side-by-side of `cube_env.py` and [src/cube.js](src/cube.js), with matching comments highlighted ("y is up, F faces +z").

**VO:**
> One thing that bit me early: the Python simulator and the 3D cube in the browser have to agree *exactly* on which sticker is which and which way is clockwise.
> If they don't, the model learns on one cube and gets tested on a slightly different one.
> So the export step writes the sticker order and the move tables into a JSON file, and the web page reads that same file. Both sides always use the same layout.

---

## 4. The neural network (4:00 – 5:30)

**VISUAL:** Diagram: 54 stickers → 324 inputs → 512-wide layer → 3 residual blocks → 12 outputs (one bar per move).

**VO:**
> Now the brain. The network takes the 54 sticker colors.
> Each color is one-hot encoded, so six colors times 54 stickers gives 324 inputs.
> That goes into a 512-wide layer, then three residual blocks, and out come 12 numbers: how likely each move is to head back toward solved.
> That's about 1.8 million parameters. Tiny by modern standards.

**VISUAL:** Code close-up of [training/model.py](training/model.py), the `ResidualBlock` class.

**VO:**
> A residual block is two layers with a shortcut around them. The block's input is added back to its output, which makes deeper networks much easier to train.

**VISUAL:** Quick montage of a git log showing the commit "training using transformer", then a later commit going back to the residual network.

**VO (optional detour, keep it short):**
> Full disclosure: I also tried a transformer, with each sticker as a token and attention learning which stickers move together.
> It was a fun experiment, but `[say what happened: e.g. it trained slower / solved fewer cubes / wasn't worth the complexity]`, so I went back to the simple residual network.
> Sometimes the boring model wins.

---

## 5. Training (5:30 – 6:30)

**VISUAL:** Terminal running `python train.py --steps 20000`, with the tqdm progress bar and the loss and accuracy numbers ticking. Speed it up in the edit.

**VO:**
> Training is plain supervised learning. Show the network a position, ask it to guess the move, and nudge its weights toward the right answer with cross-entropy loss and the Adam optimizer.
> Each step uses 160 fresh scrambles, which is over 4,000 positions. Twenty thousand steps is about 80 million examples.
> On my machine that took about 15 minutes. It uses an NVIDIA GPU or Apple Silicon if you have one.

**VISUAL:** Show the accuracy number settling at `[final accuracy]`. On-screen note: *"Accuracy looks low, and that's expected."*

**VO:**
> You might expect accuracy near 100 percent. It isn't, and that's fine.
> Remember, the labels are noisy. Close to solved, the right move is obvious. Deep in a scramble, many moves are about equally good.
> The network doesn't have to be sure. It only has to rank good moves above bad ones.

**VISUAL:** Split screen. The terminal saves a checkpoint, then the browser refreshes and loads the new model.

**VO:**
> A nice workflow detail: every 1,000 steps, the training script also exports the model for the web page.
> So I can refresh the browser *while it's still training* and watch the AI get smarter in real time.

---

## 6. Search: turning guesses into a solution (6:30 – 8:00)

**VISUAL:** Animation of a tree. The root is the scrambled cube, with 12 branches. Most branches fade out and only the top few stay bright. Each surviving branch spawns 12 more, and again only the best survive.

**VO:**
> Here's the catch: the network only gives us *probabilities* for the next move. If we always take its top pick, one bad guess sends us off course, and the cube never comes back.
> So we add search, specifically **beam search**.
> Start from the scrambled cube. Try all 12 moves and score each one with the network.
> Keep only the best few paths, say the top 1,000. Then extend each of those by every move, score them again, and keep the best 1,000 again.
> Repeat until one of the paths hits the solved state.

**VISUAL:** Code close-up of [training/search.py](training/search.py), highlighting `candidate_scores = scores[:, None] + log_probs` and the `np.unique` deduplication.

**VO:**
> A path's score is the sum of its log-probabilities, so the network's confidence adds up along the whole path.
> Two small tricks keep the beam efficient:
> First, never undo the move you just made.
> Second, different paths often land on the same position, so we keep only the best-scoring copy and free that slot for something new.

**VISUAL:** The browser's "Search width" dropdown. Solve the same scramble at 100, 1000 and 3000, showing the time and move count for each.

**VO:**
> On the page, you can choose how wide the beam is.
> A width of 100 is fast but finds longer solutions. 3,000 takes longer but finds shorter ones.
> On this scramble, 100 found a `[N1]`-move solution in `[T1]` seconds, and 3,000 found `[N3]` moves in `[T3]` seconds.

---

## 7. Running it in the browser (8:00 – 9:00)

**VISUAL:** Browser DevTools Network tab loading `solver.onnx` (about 7 MB) and `solver.json`.

**VO:**
> So how does a PyTorch model end up in a web page?
> The training script exports it to **ONNX**, a standard format for neural networks. In the browser, **ONNX Runtime Web** runs it with WebAssembly.
> The page reads the sticker colors off the 3D cube, turns them into the same 54-number state the network was trained on, and runs the exact same beam search, rewritten in JavaScript.

**VISUAL:** Code close-up of [src/solver.js](src/solver.js), the `readState` function, then the `solve` loop.

**VO:**
> No server, no API calls. Everything runs on your own machine.
> The 3D cube is built with Three.js. Every turn is animated, and after each turn the pieces snap back to the grid, so tiny rounding errors don't build up after hundreds of moves.

---

## 8. Results (9:00 – 9:40)

**VISUAL:** Terminal running `python search.py --beam 1000` on the fixed test scrambles, with "solved in N" lines scrolling past and a summary at the end.

**VO:**
> To measure it properly, I test on a fixed set of scrambles, each 100 random moves deep.
> With a beam width of 1,000, it solved `[X]` out of `[20]`, with an average solution of `[Y]` moves and about `[Z]` seconds per solve.
> For comparison, the best possible solution is never more than 26 quarter-turns. So `[comment honestly: e.g. "we're not optimal, but not bad for a 15-minute model that learned from nothing"]`.

---

## 9. Wrap-up and what's next (9:40 – 10:15)

**VISUAL:** Back to the cube solving itself one more time, slower, with the final move highlighted.

**VO:**
> So to recap:
> One: generate unlimited training data by scrambling a solved cube and recording the moves that undo the scramble.
> Two: train a small network to guess which move heads back home.
> Three: use beam search to turn those guesses into a full solution.
> And all of it runs right in your browser.

**VISUAL:** List of next steps on screen.

**VO:**
> Next up, I'd like to train a bigger model on a GPU for shorter solutions, add drag-to-turn controls, and maybe a solve timer so you can race the AI.
> The code is on GitHub, linked below. You can train your own model in about 15 minutes.
> If you enjoyed this, subscribe, and tell me in the comments what puzzle I should teach it next.

---

## Description box (draft)

```
I trained a small neural network to solve a Rubik's cube with no human solving knowledge. It learns only by scrambling a solved cube and learning to undo the moves. It runs entirely in your browser with Three.js and ONNX Runtime Web.

⏱️ Chapters
0:00 Cold open
0:30 Why the Rubik's cube is hard
1:30 The trick: scramble it backwards
3:00 Representing the cube
4:00 The neural network
5:30 Training
6:30 Beam search
8:00 Running it in the browser
9:00 Results
9:40 What's next

🔗 Links
Code: [GitHub repo link]
EfficientCube paper (Takano, 2023): https://arxiv.org/abs/2106.03157
Three.js: https://threejs.org/
ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
```

---

## Shot checklist

- [ ] Cold open: scramble, then AI solve, full screen, clean UI
- [ ] Slow 3D orbit of a scrambled cube (for section 1)
- [ ] Animation: solved cube, one move, the "undo" label (section 2)
- [ ] Exploded sticker view, then the flat 54-sticker row (section 3)
- [ ] Network diagram: 324, 512, 3 blocks, 12 (section 4)
- [ ] Sped-up `train.py` terminal recording
- [ ] Browser refresh mid-training showing the model improving
- [ ] Beam-search tree animation (section 6)
- [ ] Same scramble solved at beam widths 100, 1000 and 3000, with times and move counts noted
- [ ] DevTools Network tab loading `solver.onnx`
- [ ] `search.py --beam 1000` results on the test scrambles
- [ ] Fill in every `[placeholder]` with real numbers
