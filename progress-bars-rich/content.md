# Beautiful Progress Bars with Rich

Progress bars are a simple yet powerful way to keep users informed about a program’s progress. You’ve probably encountered tools like tqdm, but what if you need more than a basic progress bar? Have you ever seen these progress bars before when running someone else's python code?
<div style="text-align: center; width: 100%;">
  <img 
    src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGlleDc3YzFkbGVzNno4bjZwNjcyM2NnZXlrZml2cHpldTNvem8wbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/KFtYL5EQi0kkCn4to2/giphy.gif" 
    style="max-width: 100%; height: auto;"
  />
  <p style="text-align: center; font-size: 17px; margin-top: 10px; color: #555;">
    tqdm progress bar
  </p>
</div>

Maybe you've used them yourself. A lot of people use tqdm[^1] and for good reason - it's simple, lightweight, and effective. But what if you want more? If you're looking for customizable, feature-rich progress bars, the rich[^2] library is an excellent alternative.

In this post, I’ll show you how to create stunning progress bars using the Rich library, customize them for your needs, and compare their performance with tqdm.

## What is Rich?
Rich is a python library for beautiful text formatting in the terminal. It's super customizable and has a lot of built-in capability for formatting text like the ability to create tables, display markdown text, pretty print data structures, automatic syntax highlighting, and more.

It's very helpful for easily combing through a lot of data on your terminal. You can see some examples of what is possible by running `python -m rich` in your terminal after installation.
![rich-demo](https://i.ibb.co/ZzpKvnf/Screenshot-2024-12-06-at-10-59-42-AM.png)

It's easy to get started with Rich by just using the built-in `rich.print` function which will automatically pretty print whatever you want. It can be used as a drop in replacement to `print`.
```python
from rich import print

dictionary = {"a": 1, "b": True, "c": [10, 11, 12], "d": {"e": False, "f": 3.14}}
print(dictionary)
```

Rich's capabilities go far beyond just `rich.print`. Let's explore progress bars!

## Rich progress bars
To see a quick demo of Rich progress bars you can run `python -m rich.progress` in your terminal. You should see three different progress bars displayed, each progressing at different rates. The simplest progress bar you can create is `rich.progress.track`.
```python
import time
from rich.progress import track

for _ in track(range(1000)):
    time.sleep(0.01)
```

This will bring up a progress bar that looks similar to the ones shown in the demo. This is kind of like `rich.print`: It's a simple, easy to use version of something that can be further customized. Let's rebuild this basic progress bar using the building blocks provided by rich:
```python
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
)
task1 = progress.add_task("Working...", total=1000)
with progress:
    for _ in range(1000):
        progress.update(task1, advance=1)
        time.sleep(0.01)
```
<div style="text-align: center; width: 100%;">
  <img 
    src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2FlZnY1aGV6dng0OGdyNHQ2Mm16OTcwcmJhMzJ0b3k1ZjgwdXFxcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wLShucfAuvehbAgSBJ/giphy.gif" 
    style="max-width: 100%; height: auto;"
  />
  <p style="text-align: center; font-size: 17px; margin-top: 10px; color: #555;">
    Rich progress bar
  </p>
</div>

I find it helpful to think of the `Progress` class as a "view" that defines the layout and behavior of your progress bars. This is where you can customize the layout of the progress bar and include extra information like time remaining, text, etc. Above I was using the class as a container to hold my one progress bar. 

Meanwhile each individual progress bar is a called a "task" and is added to the progress view. In the above case I'm adding a single progress bar with the description "Working…" and the size of the container is 1000. The `track` feature handled advancing the progress bar for us, but now we have to handle that ourselves with the `update` method. The advantage is that we have more control over how much to advance the progress bar. The total size is 1000 and we are advancing by 1 for every iteration in the for loop, but we can choose to advance by any value like 0.5 or 2. This is especially useful if your progress bar was monitoring some non-linear process (like downloading a file and the progress bar tracks the number of bytes downloaded). The value that gets returned by the `add_task` method and stored in `task1` is just an integer that is a unique ID for the task. This allows you to manage and update multiple progress bars independently within the same `Progress` view.

## Customizing progress bars
Now that we are working with the rich progress API, let's try to customize it further. In the last section we saw that the `Progress` view is made up of columns. There are a bunch of different columns you can use and are documented [here](https://rich.readthedocs.io/en/latest/progress.html#columns).

Here's a custom progress bar I created:
```python
# import columns from rich.progress

progress_custom = Progress(
    TextColumn("[progress.description]{task.description}"),
    FileSizeColumn(),  # assumes step size is bytes
    TotalFileSizeColumn(),
    SpinnerColumn(spinner_name="christmas", finished_text="🎁"),
    BarColumn(),
    MofNCompleteColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    expand=True,
)
task3 = progress_custom.add_task("Playing hard...", total=1000)
with progress_custom:
    for _ in range(1000):
        progress_custom.update(task3, advance=1)
        time.sleep(0.01)
```
<div style="text-align: center; width: 100%;">
  <img 
    src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExeDJ2dXg5b3NzejM3djdveXN0aXpsZzMzMTJrb3JyZDZpbWZhYjN6MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/zQ1UUnZPXcL3CKxVLp/giphy.gif" 
    style="max-width: 100%; height: auto;"
  />
  <p style="text-align: center; font-size: 17px; margin-top: 10px; color: #555;">
    Christmas progress bar
  </p>
</div>

Now we've added some extra columns including a nice looking spinner column. Run `python -m rich.spinner` to see a gallery of all the spinners that you can use. I also set `expand` to True so that the columns will take up the entire width of my terminal screen.

You can also create your own column by extending the `ProgressColumn` class:
```python
class EmojiProgressColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        # No total? Just show a thinking emoji
        if task.total is None:
            return Text("🤔", style="dim")

        progress_ratio = task.completed / task.total if task.total else 0

        if progress_ratio < 0.3:
            emoji = "🐌"
        elif progress_ratio < 0.7:
            emoji = "🏃"
        elif progress_ratio < 1.0:
            emoji = "🚀"
        else:
            emoji = "🎉"

        return Text(emoji, style="bold magenta")
```

Now you have a custom column you can insert into your progress view that displays the status of your process using emojis 😃

One more thing I like to do with my progress bars is wrap a `Panel` around them to give them a polished look. What this does is draw a border around your entire progress bar. You can change the border type with the `box` parameter. Use `python -m rich.box` to see all the choices.
```python
# set up progress bar…
progress_custom = Panel(progress_custom, box=rich.box.ROUNDED)
with progress_custom:
    for _ in range(1000):
        progress_custom.update(task3, advance=1)
        time.sleep(0.01)
```

## Grouping multiple progress bars
Let's go over Rich's ability to group multiple progress bars together. You can create multiple progress bars, each with different columns and display them together on the same terminal screen. One extra step to get this working is to wrap the group in a `Live` display so they get updated together.
```python
# setup progress bars from earlier…
grouped_progress = Group(progress, Panel(progress_custom, box=DOUBLE))
with Live(grouped_progress):
    for _ in range(1000):
        progress.update(task1, advance=1)
        progress.update(task2, advance=2)
        progress_custom.update(task3, advance=1)
        time.sleep(0.01)
```
<div style="text-align: center; width: 100%;">
  <img 
    src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExYmd3aDkwOHllb2xwcGlsNmx2NWQwMW5kZ3ExZDFpb214ZmcxeG11MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3Zc1pLhcreVBGoJw93/giphy.gif" 
    style="max-width: 100%; height: auto;"
  />
  <p style="text-align: center; font-size: 17px; margin-top: 10px; color: #555;">
    Grouped progress bars
  </p>
</div>

This is useful for creating multiple views for different purposes. Like maybe a view for a progress bar and another view for some metadata about the program. An example would be an ETL pipeline that needs to process a set of CSV files. You can set up one progress bar to keep track of how many CSV files have been processed and another progress bar for counting the total number of rows that have been processed.

Now that we’ve explored Rich’s features, let’s compare its performance to tqdm.

## Does it slow my program down?
Using any progress bar is going to be slower than not using a progress bar at all due to the latency involved in writing text to a terminal console. But to see the actual differences I've set up a simple experiment to measure the runtimes between using no progress bar, using tqdm progress bars, and using Rich progress bars.
```python
# Setup code for no-bar scenario
no_bar_setup = """N = 100_000"""
no_bar_code = """
for i in range(N):
    # Simulate work
    x = i * i
"""

# Setup code for tqdm scenario
tqdm_setup = """
from tqdm import tqdm
N = 100_000
"""
tqdm_code = """
for i in tqdm(range(N), desc='tqdm loop', leave=False):
    x = i * i
"""

# Setup code for Rich scenario
rich_setup = """
from rich.progress import Progress
N = 100_000
"""
rich_code = """
with Progress(transient=True) as progress:
    task_id = progress.add_task("rich loop", total=N)
    for i in range(N):
        x = i * i
        progress.update(task_id, advance=1)
"""

def compare_progress_bars():
    timeit_number = 1000
    no_bar_time = timeit.timeit(no_bar_code, setup=no_bar_setup, number=timeit_number)
    tqdm_time = timeit.timeit(tqdm_code, setup=tqdm_setup, number=timeit_number)
    rich_time = timeit.timeit(rich_code, setup=rich_setup, number=timeit_number)

    print(f"No progress bar: {no_bar_time:.5f} seconds")
    print(f"tqdm progress bar: {tqdm_time:.5f} seconds")
    print(f"rich progress bar: {rich_time:.5f} seconds")
```
> No progress bar: 2.15476 seconds  
tqdm progress bar: 10.79213 seconds  
rich progress bar: 58.51705 seconds

The above was run on a MacBook but we should see similar differences on any machine. So we can see that the code that uses a Rich progress bar takes ~5x longer to run than the code that uses tqdm. That's a pretty noticeable difference, so should we just stick with the simpler tqdm? Well, maybe not. For one thing the test we did was pretty simplistic. The code was simulating work being done in a loop by just multiplying two numbers together and that executes very fast. But in real-world use cases it's easy to imagine that the work inside the loop will take longer to execute, so the relative cost of the progress bar updates are less significant. Also we didn't do a sweep over different values of N or `timeit_number` which might make a difference for very large or very small loop iterations. One more note is that with Rich it is possible to make less frequent updates to the progress bar thanks to the `update` method. If we choose to update the progress bar every 5 iterations in the above example then the runtime more closely matches tqdm!

So ultimately the above experiment represents the worse case scenario: maximal overhead for minimal work. In real applications the gap may be smaller. While Rich’s progress bars are slower than tqdm in simple loops, their rich (pun intended) customization options and visual appeal make them a better choice for complex workflows. 

## Wrap up

That’s all for this guide to Rich progress bars! Whether you’re building quick scripts or complex applications, Rich can help you create visually appealing, highly customizable terminal outputs. Try it out and let me know what you think! You can find the full code for this post [here](https://github.com/enerrio/jax-transformer/blob/main/rich_progress_demo.py).

<div style="text-align: center; width: 100%;">
  <img 
    src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2lzN2p0bTBxdTJyMHVja3F3Y3RzdjJsN3BocWg0OGszdWdvbHN0bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Vdc28zKlws79RAZHOg/giphy.gif" 
    alt="Complex rich progress bar" 
    style="max-width: 100%; height: auto;"
  />
  <p style="text-align: center; font-size: 17px; margin-top: 10px; color: #555;">
    Complex rich progress bar. Source: https://github.com/enerrio/jax-gpt2/blob/main/gpt/utils.py#L53
  </p>
</div>

## Resources
[^1]: tqdm: https://tqdm.github.io
[^2]: rich: https://rich.readthedocs.io/en/latest/
