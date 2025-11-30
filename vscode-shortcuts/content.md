# My Most Useful VSCode Shortcuts

In my day-to-day development work I prefer to use VSCode as my editor. Whether I'm coding or note-taking, VSCode is the workhorse of my day. There are a **lot**[^1] of online[^2] debates[^3] over editors however in the end it comes down to user preference. More recently there have also been discussions about traditional editors versus AI-assisted ones like [Cursor](https://www.cursor.com/en). I've tried Cursor and although I find its AI assistant really useful, I don't do enough side projects to justify the cost so I tend to stick with VSCode and use ChatGPT or Claude when I need to.

After several years of using VSCode, I grew tired of constantly reaching for my mouse to navigate or searching Google for how to change settings, so I decided to learn more about the built-in keyboard shortcuts. I also took the chance to explore some extensions that might make my day easier. VSCode has a very popular marketplace for extensions and anyone can build[^4] their own and share it with the world.

Since learning about the keyboard shortcuts that are available I started writing down the ones that I knew I would use the most often and tried to use them in my day-to-day work. After a month or two I got to the point where I am moving much faster than I used to and don't have to refer to my cheat sheet as often. Below are my most used keyboard shortcuts that I think are the most useful for increasing productivity.

> Note: I'm using VSCode shortcuts for MacOS. However similar shortcuts are available for other operating systems as well.

## Navigation
These shortcuts are mainly for moving around the different windows within VSCode.

| Shortcut (Mac)            | Action                                                          |
|---------------------------|-----------------------------------------------------------------|
| `⌘ + B`                   | Toggle Sidebar visibility                                             |
| `⌘ + J`                   | Toggle Bottom panel                                           |
| `⌘ + Shift + E`           | Go to Explorer tab                                              |
| `⌘ + Shift + F`           | Go to Search tab                                                |
| `^ + Shift + G`           | Go to Git tab                                                   |
| `⌘ + Shift + X`           | Go to Extensions tab                                            |
| `⌘ + \`                   | Open current file in side window. Good for side by side editing |
| `⌘ + 1 or ⌘ + 2 or ⌘ + 3` | Switch between open editors in side by side editing mode        |
| `⌘ + ,`                   | Open user settings                                              |
| `F12`                     | Go to variable definition                                       |

## Terminal
The integrated terminal is a built-in terminal window available within VSCode. It's most useful if you want to run terminal commands from within your workspace or to run scripts. These shortcuts are super helpful for working with it.

| Shortcut (Mac)        | Action                                            |
|-----------------------|---------------------------------------------------|
| ``^ + ` ``              | Open/close terminal                               |
| ``^ + Shift + ` ``      | Open new terminal                                 |
| `⌘ + 1`               | Switch focus to editor                            |
| `⌘ + Shift + ]`       | Go to next active terminal                 |
| `⌘ + Shift + [`       | Go to previous active terminal                    |
| `⌘ + \`               | Open new terminal in side-by-side mode            |
| `⌘ + Option + ←`      | Move focus to terminal on the left                |
| `⌘ + Option + →`      | Move focus to terminal on the right               |

## Breadcrumbs
Breadcrumbs[^5] are a graphical navigation concept commonly used on web pages and other UI software. VSCode has its own version that lies horizontally across the top of an open file to show the path from the root of the workspace to whatever symbol your cursor is currently pointing to. 

| Shortcut (Mac)  | Action           |
|-----------------|------------------|
| `⌘ + Shift + .` | Open Breadcrumbs |

The above shortcut will open the breadcrumb viewer and then you can use the up and down arrows to jump through the symbols in your current file. You can even use the left and right arrows to navigate up and down your workspace tree and go to a different file. I find it most useful for jumping around from place to place within my currently open file.

## Editing
These shortcuts are best for making editing and writing easier.

| Shortcut (Mac)  | Action                                                                                         |
|-----------------|------------------------------------------------------------------------------------------------|
| `⌘ + X`         | Cut current line                                                                               |
| `⌘ + U`         | Go to last cursor location                                                                     |
| `^ + G`         | Go to specific line (type in the line you want to go to after hitting the shortcut)            |
| `⌘ + L`         | Select current line                                                                            |
| `⌘ + Shift + L` | Select all occurrences of current selection                                                    |
| `⌘ + Shift + O` | Go to symbol in current file                                                                   |
| `Shift + ⌥ + F` | Format entire file (you might have to configure a formatter based on the language of the file) |
| `⌘ + Shift + V` | Preview Markdown file (use only if your currently open file is Markdown)                       |
| `⌘ + K, V`      | Open Markdown file in edit and preview mode side by side                                       |

## Miscellaneous
These next few shortcuts are ones I didn't see fit into the above categories but that I still find really useful for general productivity.

| Shortcut (Mac)           | Action                                  |
|--------------------------|-----------------------------------------|
| `⌘ + K, M`               | Change language mode of open file       |
| `⌘ + K, ⌘ + T`           | Change color theme                      |
| `⌘ + F`                  | Search within current file              |
| `⌘ + Option + R`         | Toggle regex search in find widget      |
| `⌘ + P`         | Go to File      |
| `⌘ + Shift + P`         | Show command palette      |


## Bonus: Snippets
Aside from keyboard shortcuts, there are also other parts of VSCode that can improve your workflow. A couple of them are Snippets and Extensions. VSCode Snippets[^6] are custom templates that make it easier to reuse common code. Some snippets are open sourced like this [PyTorch snippet extension](https://marketplace.visualstudio.com/items?itemName=SBSnippets.pytorch-snippets) but you can make your own as well by editing a language specific snippet file[^7]. Here's my custom Python snippet file which has a single entry for the common `if __name__ == ` code block.
```json
{
    "if(main)": {
        "prefix": "ifmain",
        "body": ["if __name__ == \"__main__\":", "    ${1:pass}"],
        "description": "Code snippet for a `if __name__ == \"__main__\": ...` block"
    },
}
```
All I have to do is type in `ifmain` and then hit enter for the rest to autofill.

## Bonus: Favorite Extensions
Shortcuts are great but extensions can also greatly speed up your workflow and generally make development more enjoyable. Here are some extensions I end up using often.

| Extension Name                                                                                                  | Purpose                                                                                          |
|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)                    | Automatically generates Python docstrings                                                        |
| [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) | Spell checker. Never have embarrassing typos in your code again                                  |
| [CodeSnap](https://marketplace.visualstudio.com/items?itemName=adpyke.codesnap)                                 | Take beautiful looking screenshots of your code for easy sharing                                 |
| [indent-rainbow](https://marketplace.visualstudio.com/items?itemName=oderwat.indent-rainbow)                    | Colors the indentation level. Never get confused with what indent level you're on again          |
| [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)                               | Jupyter notebook integration                                                                     |
| [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)           | Feature rich Markdown support accessories                                                        |
| [Peacock](https://marketplace.visualstudio.com/items?itemName=johnpapa.vscode-peacock)                          | Change the color of your workspace. Makes it easier to tell apart multiple open workspaces       |
| [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)                                  | Ruff linter and formatter for Python                                                             |
| [ChatGPT – Work with Code on macOS](https://marketplace.visualstudio.com/items?itemName=openai.chatgpt)         | Easily ask ChatGPT questions about your active editor (requires a separate ChatGPT subscription) |


## Share your favorites
I've noticed that I take my hands off my keyboard less frequently thanks to using these shortcuts (although I occasionally have to look at my cheat sheet to remember some). [Let me know](https://enerrio.bearblog.dev/about-me/) what your favorite shortcuts or extensions are!

## Links
[^1]: Vim over VSCode: https://www.reddit.com/r/vim/comments/lvh6e4/what_are_your_reasons_to_use_vim_over_something/
[^2]: Vim vs VSCode: https://www.doc.ic.ac.uk/~nuric/posts/coding/why-i-switched-from-vim-to-visual-studio-code/
[^3]: Hackernews debate on Vim vs VSCode: https://news.ycombinator.com/item?id=30841460
[^4]: VSCode Extensions: https://code.visualstudio.com/api/get-started/your-first-extension
[^5]: Breadcrumb: https://en.wikipedia.org/wiki/Breadcrumb_navigation
[^6]: Snippets: https://code.visualstudio.com/docs/editing/userdefinedsnippets
[^7]: Custom snippets: https://code.visualstudio.com/docs/editing/userdefinedsnippets#_create-your-own-snippets
