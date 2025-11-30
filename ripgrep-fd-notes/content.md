# Notes on ripgrep and fd

I consider myself pretty comfortable working in the command line. I've spent a lot of time in there working as a data scientist and data engineer. Whether it's doing ad hoc data analysis on a remote EMR or running scripts on my local machine, I've gotten plenty of practice with the basics. 

I've been aware that the world of CLI tools + terminal setups is very big but I've avoided diving into it because the tools I was familiar with were good enough for getting the job done. But earlier this year I saw a colleague breeze through some complicated tasks involving searching a large amount of files using a regex pattern. All in the terminal, and very fast. At the same time, I was starting to work on a project that involved a lot of time in the command line. I kept finding it difficult to do what I wanted with my current skillset. So I resolved to learn a bit more about some new tools (and become more experienced with my usual ones like `less`) in an effort to become better at devops work.

So I searched around for some popular CLI tools and came across a couple tools I've since come to use a lot: ripgrep and fd.

## What is ripgrep?
[Ripgrep](https://github.com/BurntSushi/ripgrep) is basically a modern version of grep. It's recursive by default and faster than most grep-like tools (including grep itself). It has a lot of features but I mostly use it to do basic recursive searches in a directory. For example, I'll want to see how many times `<keyword>` is present in my entire repo so I'll run `rg <keyword> ~/<myRepoName>` which will show me all the times it's been found. 

![first](https://i.ibb.co/YSPsMbB/first.png)

A very useful flag I often use is the `-C` flag (short for  `--context`) which shows additional lines before and after the line where your search pattern was found. In the screenshot below I request the 2 lines that appear both before **and** after the found line to show up in the output.

![second](https://i.ibb.co/PGQXZCfd/second.png)

It also respects any `.gitignore` files by default although you can disable that with the `--no-ignore` flag.

## What is fd?
[fd](https://github.com/sharkdp/fd) is a simple and fast alternative to `find`. It also colorizes the output and respects any `.gitignore` files by default. You can use regex patterns like ripgrep, and filter by filetype and extension like the regular `find` command. The big selling point here is that it is generally faster than `find`. Here's a sample directory tree:

![third](https://i.ibb.co/Tz98nv1/third.png)

And I can search for files that have a specific extension like python files:

![fourth](https://i.ibb.co/C3s3g3nZ/fourth.png)

Or for directories that match a given regex pattern:

![fifth](https://i.ibb.co/BKgZ2956/fifth.png)

## Cheat Sheet
Below is my personal cheat sheet I keep in my notes that contains my most frequently used rg and fd commands. Some of them I've used so often that I can just remember them without having to refer to my notes but I like to keep them there anyway.

| `rg` Command                                   | What it does                                                                                 |
|-------------------------------------------|----------------------------------------------------------------------------------------------|
| `rg foobar`                               | recursively search for foobar                                                                |
| `rg foobar <filename>`                    | search within a single file                                                                  |
| `rg foobar <dir>`                         | recursive search within a specific directory                                                 |
| `rg -. foobar <thing>`                    | search hidden files and dirs                                                                 |
| `rg -u foobar`                            | disable .gitignore in search                                                                 |
| `rg -uu foobar`                           | search hidden files and dirs (same as `-.` flag)                                                       |
| `rg -uuu foobar`                          | search everything! binary files too                                                          |
| `rg --stats <pattern>`                    | show results and stats like time spent searching, bytes, etc                                 |
| `rg -i fast`                              | ignore case                                                                                  |
| `rg -w fast`                              | search for exact word matches                                                                |
| `rg -c fast`                              | count number of line matches                                                                 |
| `rg -F def(in`                            | disable regex                                                                                |
| `rg fast -C2`                             | show 2 lines before and after every match                                                    |
| `rg -a fast`                              | search binary files                                                                          |
| `rg -z fast`                              | search compressed files                                                                      |
| `rg foobar -g '*.py'`                     | search all files that match glob pattern                                                     |
| `rg foobar -g '!*.py'`                    | search all files that __don’t__ match glob pattern                                           |
| `rg foobar -tpy`                          | search all files that end with py                                                            |
| `rg foobar -Tpy`                          | search all files that __don’t__ end with py                                                  |
| `rg precision -tpy -l`                    | search python files but only show the filepaths, not the content                             |

---

| `fd` Command                                   | What it does                                                                                 |
|-------------------------------------------|----------------------------------------------------------------------------------------------|
| `fd <pattern>`                            | search recursively for files/directories that match the pattern                              |
| `fd -e py`                                | find anything with a given extension                                                         |
| `fd -t f <filePattern>`                   | find only files                                                                              |
| `fd -t d <dirPattern>`                    | find only directories                                                                        |
| `fd -e py --max-depth <int>`              | find anything within a certain depth                                                         |
| `fd -e ipynb . --changed-within 10days`   | find anything changed within a certain time frame                                            |
| `fd -e py -0 <pattern>`                   | find files and use null character to delimit results instead of newline                      |
| `fd -e py -0 <pattern> \| xargs -0 du -h` | find files/dirs and pipe to xargs which runs a command on each result                        |
| `fd -e py -x du -h`                       | same as above but more concise. run another command on each result                           |
| `fd -e py -X echo {}`                     | similar to above but runs another command once on all results. Batches results into one call |


## Bonus: eza
[Eza](https://github.com/eza-community/eza) is an awesome replacement for `ls` that includes color coding, icons for files/folders, and more. The color coding is especially useful when you're listing a folder with a lot of items in it. I like it so much that I've aliased `ls` to `eza --icons=always` so that the icons always show up.
