# Dispatch from the Frontier - Adventures in AI Engineering

Anthropic released Claude Code, a terminal-based agentic coding tool, about a year ago. Since then, it has been gaining steam among software engineers for its coding capabilities, amplified with each release of Sonnet and Opus models. It has been widely adopted by individual developers and large organizations alike.

It reached an inflection point last December with the release of Opus 4.5. The model itself was a big jump in capabilities and was even heralded by some as AGI[^1]. I had used Claude Code a bit last year, but with the more recent releases of Opus 4.6 and Sonnet 4.6, I wanted to give it a try on a new project. I was especially interested to see how it could handle long-running autonomous coding sessions. My goal was to see whether a new project could be spun up from scratch with minimal intervention. In this post, I'll talk about the techniques I used and whether this experiment was a success or not.

![claude-code](https://i.ibb.co/WNtt38PM/cc.png)

## Claude Code Intro
Before diving into the project, I want to cover some aspects of Claude Code. There are two main features I want to cover that are important for this experiment:

1. Context limit
2. Permissions

No matter what model you are using, there is a limit to the amount of context Claude can hold in its session. Both the latest Opus and Sonnet models have a context window of 200k tokens (with a 1 million token window in beta). Making proper use of the context is critical to getting good results. If you go over that limit, then compaction happens, which is when the entire conversation gets summarized and carried over into a new session. Important information can be lost during this stage so it's preferable to limit the scope of a chat session to a single task or at least have some kind of external markdown file to act as a "memory" for the agent to read and write to across sessions. The technique I use later focuses on short sessions to avoid compaction.

By default Claude will ask for permission to perform certain actions like reading from a directory, editing a file, running bash commands, etc. You can bypass these asks and let Claude do whatever it wants by running in a "headless" mode. This can be enabled by launching Claude Code with the `--dangerously-skip-permissions` flag set. I use this feature to let Claude work uninterrupted.

One more note about permissions: since this experiment involves letting Claude use tools unfettered, I found it crucial to set up [proper hooks](https://code.claude.com/docs/en/hooks-guide) that restrict Claude from performing destructive actions on my machine. Hooks are small pieces of code that run at certain points within the agentic loop. These protective hooks do things like prevent Claude from running `rm -rf` on my root directory and the like.

Claude Code is not the only agent in town. Their primary competitor is [OpenAI's Codex](https://openai.com/codex/) with their specialized coding models (sorry [Gemini CLI](https://geminicli.com)). It's debatable which is better. However, I chose Claude Code because I've come to prefer its approach to planning and execution. I tried Codex CLI and was prepared to use it when I inevitably hit Claude's usage limits, but it turned out not to be necessary because Claude finished the project in a reasonable amount of time.

## Project Idea
Let's take a look at how I set up this project. First off: what is the project?

I had recently read some articles about `git log` and `git worktree` in an attempt to improve my git skills. I had also asked Claude to create some git-based exercises I could work through to reinforce what I learned and brush up on some more advanced git topics as well. I decided to expand on that exercise set and create an open source terminal-based set of exercises that anyone can go through to learn git. I also really like [Rustlings](https://rustlings.rust-lang.org), the set of exercises for learning Rust, and wanted to emulate that setup.

I had a vision for what this project should look like but now came getting Claude to code it all. There are many approaches I could have gone with. Some people do the coding themselves and use Claude as a substitute for Stack Overflow, some have it write pieces of code and spend most of their time reviewing and editing Claude's generations. But I wanted to do something a little more extreme.

I wanted to try a technique called "ralph loops" to give full control to Claude while using small-scoped sessions to avoid compaction.

You can find the completed project on GitHub here: [https://github.com/enerrio/gitgym](https://github.com/enerrio/gitgym)

## Ralph Loop
Last summer Geoffrey Huntley introduced the idea of [Ralph Loops](https://ghuntley.com/ralph/). For a full overview, you can read the linked article (and [this one](https://ghuntley.com/loop/)), but a simplified explanation is that it's essentially made up of 2 simple concepts:

1. Spec-driven development
2. A while loop

By spec-driven development, I mean writing in plain English what you want Claude to do and being comprehensive in the requirements. You don't have to go into the minutiae of the implementation details, Claude can figure it out, but you should be clear in the high-level objective of what you want to do and how Claude should behave when acting. An important caveat: you should not write these specs yourself! Let Claude write them for you. You can make your intentions clear with conversations between yourself and Claude (preferably Opus for this planning stage).

After all the specs are written and it's ready to implement, then Claude can go ahead and start implementing (preferably using Sonnet since it's cheaper and you'll get more mileage if you're on a subscription plan). The while loop is essentially piping the same prompt into a headless Claude and letting it complete tasks one at a time. This prompt will say something like "study the specs, implement the first uncompleted task, mark as completed when you're done, then exit." This continues forever until all tasks are completed and then the loop should exit.

I'll walk through how I did this in the next few sections and include transcripts of my prompts.

This technique is probably most useful for greenfield projects like this one. The Ralph loop excels at autonomous coding, but maintenance of existing projects likely requires either careful prompting or different lines of thinking altogether. Letting it run loose could be destructive to existing code e.g., Claude deleting some function because it thinks (incorrectly) it's unnecessary.

## The Interview
I ended up following a version of Huntley's [YouTube video](https://www.youtube.com/watch?v=4Nna09dG_c0) and the [Ralph playbook](https://claytonfarr.github.io/ralph-playbook/) by Clayton Farr. The first step was a conversation with Opus to fully flesh out the project idea and generate a specification file. An important part was instructing Claude to not start building anything. This is just a planning phase where Claude basically interviews me about what I want this project to look like.

My initial prompt looked like this:
> I want to build an interactive platform to learn git, from basics to advanced stuff. The platform should take inspiration from Rustlings which is an interactive platform for learning rust. It should cover important aspects of git, from basics like git clone and git pull to git rebase and git log --oneline --graph --all, etc. It should also include quizzes for people to test their knowledge. It should be open source. It should have them run the actual commands but in a safe way so that they're not messing with real repos or with their own local stuff.
> 
> Don't build anything yet. Interview me about requirements one question at a time to understand the edge cases, configuration needs, and expected behavior. After we've covered enough ground, generate a specification document at specs/README.md.

From there Claude asked me a series of questions and ended with creating the `specs/README.md` file.

I've uploaded the full transcripts of my chats in a separate repo: [https://github.com/enerrio/gitgym-prompts](https://github.com/enerrio/gitgym-prompts). Be sure to check it out for more info.

## Scaffolding
Now that I had a spec, I still needed a few more files. I used Claude to read the spec and generate an implementation plan that will be used as a sort of "state file" or "memory" for Claude to keep track of its progress and know what to do. This `PLAN.md` file is detailed and describes step-by-step what Claude should do. It's ordered as well, so Claude should start at the top and work its way down.

Snippet of `PLAN.md`:
> # gitgym - Implementation Plan
> 
> Each task is small and independently testable. Complete them in order; check off each box when done.
> 
> ---
> 
> ## Phase 1: Project Scaffolding
> 
> [ ] Initialize a `uv` project: run `uv init`, set up `pyproject.toml` with project metadata, Python >=3.12, `click` dependency, `[project.scripts] gitgym = "gitgym.cli:main"`, and hatchling build backend  
> [ ] Create the source package directory `src/gitgym/` with `__init__.py` (version string) and `__main__.py` (`from gitgym.cli import main; main()`)  
> [ ] Create `src/gitgym/config.py` with path constants: `WORKSPACE_DIR = ~/.gitgym/exercises/`, `PROGRESS_FILE = ~/.gitgym/progress.json`, `EXERCISES_DIR` (package-relative `exercises/` directory)  
> [ ] Create the empty `exercises/` directory at project root (will hold exercise definitions, shipped with the package)  
> [ ] Configure `pyproject.toml` so that `exercises/` is included in the built package (hatchling `[tool.hatch.build]` settings)  
> [ ] Run `uv sync` and verify `uv run gitgym --help` prints output without error
> 
> ## Phase 2: Exercise Loading
> ...

Another file I needed was a basic `prompt.md`, the prompt that is fed into Claude on each iteration of the loop. It should be short and simple. It provided basic background on the project and instructed Claude on what it should be doing for the current session which is basically: review the specs, look at the first unfinished task, complete it, add tests to confirm it works, mark it as complete in the `PLAN.md`, and exit. It was important to clarify that Claude should not be trying to do everything at once and should only work on one task at a time. Otherwise it risks filling its context window and triggering compaction, which can degrade performance.

My `prompt.md` file:
> You are working on the gitgym project.
> 
> ## Instructions
> 
> 1. Read specs/README.md (the Pin) to understand the full specification.
> 2. Read specs/PLAN.md to see the current implementation state.
> 3. Pick the FIRST task marked [ ] (not started). Do NOT skip ahead.
> 4. Implement ONLY that single task. Do NOT combine multiple tasks.
> 5. Follow existing patterns in the codebase. Use search to find how similar things are already done before writing new code.
> 6. Write tests for the task. Run them. Fix any failures.
> 7. If all tests pass, commit with a descriptive message referencing the task number (e.g., "task 2.1: add config module with workspace paths").
> 8. Update specs/PLAN.md: change [ ] to [x] for the completed task.
> 9. Stop. Do not continue to the next task.

Finally, I needed an executable script that contains the actual loop. This can be as simple as `while true; do cat prompt.md | claude --print --dangerously-skip-permissions; done`. However, I wanted a little more functionality so [my script](https://github.com/enerrio/gitgym-prompts/blob/main/run_loop.sh) ended up looking a little more complicated. The spirit of the script is the same: while loop over Claude Code. But I wanted to be able to switch to Codex in case I hit usage limits. I also wanted the transcripts to be saved to disk for auditing purposes, and allow it to exit early if all tasks are complete or no progress is being made.

## Let It Rip
With all that in place, I was ready to let Claude loose. I wanted to keep an eye on it at first, so I just let it run for 5 loops to start. The logs looked good and I could see the code being generated and the `PLAN.md` getting updated. It seemed to be doing well and I had not hit any limits yet, probably because the cheaper Sonnet was being used for coding. I restarted the loop and let it go for ~25 iterations or so.

After it completed, about half of the tasks in the `PLAN.md` file were done and I was able to run `uv run gitgym --help` and see some results. At this point I started a new session with Opus to have it inspect the codebase and the progress made so far to get its opinion. I wanted to see if there were any deviations from the plan or if there was anything in the code that didn't match the spec. Opus had concluded that everything looked good, so I continued the loop. At this point, however, I exhausted my usage limits and had to wait a day before restarting the loop.

The following day, I restarted the loop for longer and just let it run. I took a look at the logs from time to time but largely let it continue on its own. It would be too time-consuming to double-check everything Claude was doing.

Eventually Claude had completed all the tasks and I was happy with the results. I went through all the exercises to make sure everything was working well. I liked the exercises. They were educational and accurate, and I ended up learning some things I didn't know before. I do think it is a genuinely useful hands-on intro to git. 

The tool had worked as expected but I did have some notes.

## Finishing Touches
There were a few minor things that I prompted Opus to fix like a more comprehensive README, clearer exercise names, and an extra command to clean the workspace so the exercises don't pollute a user's machine. Opus made these changes and also helped walk me through the process of setting up CI through GitHub Actions and publishing the package on PyPI (I had not done this before so it was very educational for me as well).

One thing I did have to go back and forth with Claude on was the CI integration. The CI workflows passed locally but failed in Github. After some debugging, Claude identified that one error was due to an [incorrect symlink](https://github.com/enerrio/gitgym/commit/daff591a53fd975d1182e26b7f38e1b97fe75f54) and another was due to the [default git branch](https://github.com/enerrio/gitgym/commit/917e6d9427c6741df1dd55be4ae3c51c9ffb2062) being "master" instead of "main." Even though the autonomous coding was successful, some manual intervention was required.

## The Future

All in all, I was very happy with the results of this experiment and consider it a success. The git learning tool worked well for its intended purpose and leaves the door open for additional exercises. It also didn't cost me anything beyond the $20 for the pro subscription plan. When I hit a usage limit, I would just kill the loop script and restart it the next day. Since the tasks are tracked in an external `PLAN.md` file and each Claude session is scoped to a single task, nothing is lost between sessions. Most sessions aren't even filling up the context window.

Wes McKinney (creator of Pandas) wrote [an article](https://wesmckinney.com/blog/mythical-agent-month/) about working with agents and the risk of accruing more and more technical debt. This is not only possible, but likely if not enough attention is paid to the architectural decisions being made by the model. I did not pay much attention to the code Claude was producing and pretty much left it to its own devices, although I did notice that it ended up writing a huge number of tests (836!). I suppose this is a good thing because the agent will get useful feedback if it breaks something, but it's definitely more than a human would've constructed.

This was a simple program, so it's difficult for me to say if my approach would scale well to larger projects, although it would not surprise me if others are attempting to do so. When I've used Claude Code (and other AI tools) in my other projects, I've kept a more watchful eye on the outputs. I spent more time reviewing and understanding the code that is generated. In both scenarios though, I hardly wrote any code myself, which is a huge shift from just a couple years ago. We've truly entered a new world with these tools. But despite the superpowers they grant engineers, I believe it will be important to stay grounded and not give up full control. We may feel empowered, but we should keep one foot rooted in reality and keep our expectations in check about what these models are capable of. That may be difficult since they are already extremely capable and getting more so with each release (especially when you consider what they can do beyond writing code). Maybe one day they will be able to operate on [longer time horizons](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) and larger codebases. For the time being, it will be crucial to have a human in the loop, monitoring progress and steering the model. This was well discussed in Armin Ronacher's article on [agent psychosis](https://lucumr.pocoo.org/2026/1/18/agent-psychosis/). Only in hindsight will we know if the models of early 2026 truly supercharged us or if we're falling down a rabbit hole. The most important skill may be knowing when to apply discipline and when to let it rip.

## Resources
[^1]: Dean Ball Tweet: [https://x.com/deanwball/status/2001068539990696422](https://x.com/deanwball/status/2001068539990696422)
