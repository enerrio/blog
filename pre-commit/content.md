# Why Pre-Commit Hooks Are Essential (Even If You Hate Them At First)

What is pre-commit and why it is so useful in a project?

Wait, come back! 

Maybe you've come across it in an open source library or seen it at your job and cursed whoever put it there. The first time you encounter pre-commit, it can be annoying. You've just made a change and want to commit it but get hit with a long error message instead:

```bash
$ git commit -m "minor change to main.py"
check yaml...........................................(no files to check)Skipped
fix end of files.........................................................Failed
- hook id: end-of-file-fixer
- exit code: 1
- files were modified by this hook

Fixing README.md

trim trailing whitespace.................................................Failed
- hook id: trailing-whitespace
- exit code: 1
- files were modified by this hook

Fixing main.py

ruff (legacy alias)......................................................Passed
ruff format..............................................................Failed
- hook id: ruff-format
- files were modified by this hook

1 file reformatted
```

Not only did your changes fail to commit, but files were auto-edited so now you have to stage and commit again. What gives!? It takes some getting used to, but in the long run pre-commit helps standardize your codebase's formatting and keep out bugs. In this post I'll cover what pre-commit is and why you should be using it in your projects.

## What's it good for?
Pre-commit is a software tool that, like the name implies, runs just before you try to commit something to git. It invokes so-called "hooks" which are essentially small programs. For example there are hooks that can [validate yaml](https://github.com/pre-commit/pre-commit-hooks/blob/main/pre_commit_hooks/check_yaml.py), [remove trailing whitespace](https://github.com/pre-commit/pre-commit-hooks/blob/main/pre_commit_hooks/trailing_whitespace_fixer.py), and even execute your project's test suite to ensure that whatever you're committing won't break anything. Because hooks are just standalone programs, you can do almost anything with them. You can even write a hook that asks an LLM to critique your changes and give a pass/fail score (please don't do this)[^1].

The benefits of this type of standardization is that everyone contributing to the project is on the same page with respect to formatting, style, and testing. Hooks can be a powerful addition to your project and help you focus on what is important to your project without getting slowed down by the little things.

## How to set it up
First make sure you have the pre-commit tool installed on your machine. The full installation instructions can be found [here](https://pre-commit.com/#install).

Next is setting up the `.pre-commit-config.yaml` file. The config file defines where the hooks live (often this is just a Github url but it can also be `local` for custom hooks), the name of the hook to use, and any extra settings you want to apply to the hooks. Here's an annotated `.pre-commit-config.yaml` file:

```yaml
repos:
  # Repo containing source code for the check-yaml, end-of-file-fixer, and trailing-whitespace hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    # Which tag to checkout from source repo
    rev: v6.0.0
    # A list of hooks to use for this project. Each hook is defined in the source repo
    # The full list of available hooks in the source repo in defined in `setup.cfg` which maps the ids to hook's code
    hooks:
    -   id: check-yaml
    -   id: end-of-file-fixer
         # Exclude a fixed-width data file - it must not end with a newline
         exclude: '^data/fixed_width_records\.txt$'
    -   id: trailing-whitespace
  # Another repo containing code formatting hooks
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.0
    hooks:
      - id: ruff
      - id: ruff-format
  # Custom local hook - defined locally at hooks/llm_check.py
  - repo: local
    hooks:
      - id: llm-check
        name: llm-check
        # The command that this hook should run. In this case it's a python script
        entry: python hooks/llm_check.py
        language: python
        # This script depends on a 3rd party library
        additional_dependencies: ['anthropic>=0.39.0']
        # Additional settings
        always_run: true
        pass_filenames: false
        verbose: true
```

Now you must install the hooks before they can be used.
```bash
pre-commit install
```

Don't forget to do this or else the hooks won't run! Once they're downloaded, the hooks will run whenever you try to commit something. It might be a little slow the first time you run it because there's some setup that happens, but that's normal. Future commits will be faster.

You can control which files should be targeted by a hook with the `include` and `exclude` arguments. For example if you have a fixed length file with some specific formatting required by your code then you can configure the `end-of-file-fixer` pre-commit hook so that file is excluded. That's exactly what is happening in the example above.

Many programming languages are supported so a hook can be written in Python, Rust, and [many others](https://pre-commit.com/index.html#supported-languages). This is completely transparent to you as a user though. It doesn't matter what language is used, pre-commit will execute it on your behalf. An important thing to consider however, is that some hooks might be slow if they're doing a lot. Sometimes this can be amplified by the language itself (like python). But for the most part you shouldn't worry about this. Most of the time the hooks created by the pre-commit team are the ones you'll use most often, and they are already fast. Something to keep in mind if you are writing your own custom hooks.

There's a whole host of options available for the config file. Check out the documentation [here](https://pre-commit.com/index.html#pre-commit-configyaml---hooks).

Now that you're familiar with pre-commit, let's talk about why your team will thank you.

## Human Collaboration...
Having a standardized set of hooks can eliminate friction within a team of engineers. No more one-off pull requests to "fix formatting" or disagreements over linting issues. Removing this friction allows you and your teammates to focus on the important parts of your project.

Let's say there's some emergency and you need to push out a change fast but you don't have time to fix the issues that pre-commit surfaces. So what do you do? Don't worry, you are not stuck. For these situations pre-commit has a [skip feature](https://pre-commit.com/index.html#temporarily-disabling-hooks) so you can skip one or more hooks. To skip them all you can add a `--no-verify` flag to your commit command. For example:
```bash
git commit -m "emergency change" --no-verify
```

## ...And Beyond
Not only are these important for your teammates, but they can also be helpful to AI assistants. In today's world AI-powered coding assistants are becoming more and more commonplace. Anthropic's [Claude Code](https://code.claude.com/docs/en/overview) and OpenAI's [Codex](https://openai.com/codex/) are two of the most powerful agentic coding tools on the market. And every month it seems the frontier of what these tools are capable of is pushed further.

AI tools can use pre-commit hooks so that their changes follow the same standards as human contributors. With sufficient hooks an AI agent can make changes, see what errors pre-commit surfaces, and then go about fixing them on its own. Think of this as one type of guardrail to keep slop from entering your codebase. Other measures should be taken but this is an easy one to set up.

## Pro Tips
1. Don't forget to install the hooks! Make sure to run `pre-commit install` when you first set up a repo.
2. If you want to run the hooks outside of `git commit` then run `pre-commit run --all-files`. To run an individual hook do `pre-commit run <hookID>`.
3. Run `pre-commit autoupdate` to update your hooks up to their latest tag. This will edit the `.pre-commit-config.yaml` file.
4. Set `fail_fast` to true for long-running hooks. When pre-commit runs, all installed hooks run by default. If you want avoid that and have it exit after the first failure occurs then this is the setting for you. This can save time if you have some hooks that take a bit to finish.
5. If you include a hook for running a test suite, configure it to run only tests that take seconds to finish. That way you aren't waiting for long-running tests to complete on every commit.

## Bonus
If you’re interested in a custom hook that can drain your API credits, check out this `llm-check` hook included in the [GitHub repository](https://github.com/enerrio/pre-commit) that accompanies this post. Yes, it runs an LLM during your commit.

![llm-check](https://i.ibb.co/39L9ZGSh/llm-check.png)

## Bye
I hope I've convinced you that pre-commit can be a boon to your projects. These are just the basics. There's a ton more you can configure so that pre-commit is perfect for your project. Have a favorite hook? Built a custom one for the agentic coding era? Reach out and let me know!

## Resources
[^1]: A hook like this will burn through tokens on every commit and take time for a response from the LLM provider, wasting time and money. Plus not every commit should be subjected to a pass/fail test. 
