# Social Media Bots

Social media bots can be both a blessing and a curse - some flood communities with spam and scams[^1], while others provide valuable public services, such as sending alerts when earthquakes occur[^2]. Then there are bots that are somewhere in between. The fun ones. Like a bot that posts about adoptable pets[^3] or one that posts Merriam Webster's word of the day[^4]. I made the latter last year for [Bluesky](https://bsky.social/about), the popular social media site. One of the advantages of Bluesky is that it has a developer API for people to build their own bots, feeds, and clients. A feed is kind of like a custom algorithm that collects content and delivers it to the feed's subscribers. You can imagine having a feed for different topics like a film buffs feed that aggregates posts about the movie industry. Clients are applications that can act as alternatives to the main Bluesky website. Instead of viewing posts on bsky.app you can access them through an alternative website that presents the same content with a different visual interface. In this post I'll talk more about a few bots I built and how I deployed them to run daily.


## A wordy bot
Let's go back to that word of the day bot. [Merriam Webster](https://www.merriam-webster.com) selects a random word as "word of the day" and displays it on their homepage. They give a link to the word's definition, etymology, pronunciation, part of speech, and more. If you give your email they'll even send you the word of the day directly to your inbox. Last year I joined Bluesky and noticed a few bots that I thought were cool (like the adopt a pet bot). I wondered what it would take to create my own bot (how to schedule it to post automatically, where to get data, etc) and thought it would make a good weekend project to learn some new skills. I found Bluesky's developer docs and saw they had a nice section[^5] on getting started with bots. 

Next was figuring out how it would all come together. I initially thought I could scrape Merriam Webster's homepage for the word of the day but then discovered that they also have their own API[^6]. I can use this API to directly get the word of the day, clean it up, and create a Bluesky post out of it. But there were a couple challenges to consider:

1. The API returned multiple definitions for the word of the day and had some extra API-specific information in it. That means the actual definition would have to be extracted and cleaned.
2. How can the actual posting be automated?

For the first point, it just took some trial and error experimenting with various regex patterns (and several sample words) to find one that reliably extracted the definition and nothing more. The response from the API can look a little intimidating. Here's a snippet of the response for the word "rapport":
```
…
"def": [
    {
        "sseq": [
            [
                [
                    "sense",
                    {
                        "dt": [["text", "{bc}a friendly, harmonious relationship"]]
                    }
                ]
            ]
        ]
    }
]
…
```
The text we care about is "{bc}a friendly, harmonious relationship". You can see there is a "{bc}" tag that doesn't really make any sense outside the context of the API. Without getting too deep into the weeds of the Webster API, definitions often include special formatting tags that need to be removed before posting. This can be done with a [regex pattern](https://regex101.com), and to make it reliable I collected the API responses for a bunch of words to ensure that the final pattern was able to clean all of them.

Posting to Bluesky itself was straightforward. First I created an account specifically for the bot, then using Bluesky's API I can login to that account using its login credentials and call a function that would create a post.

Finally there came deployment. I had some familiarity with AWS Lambda and EventBridge so I opted to go that route. I packaged my code as a [Lambda function](https://aws.amazon.com/lambda/) and configured all the necessary components: increased timeout slightly, set up proper logging, added Bluesky login credentials as environment variables. Then I set up an [EventBridge rule](https://aws.amazon.com/eventbridge/) that would trigger the Lambda function according to a cron schedule (daily at 10 AM Pacific time). I'm on the free tier of AWS so I didn't have to worry about cost at all since the free tier allows for a large volume of function calls before charging.

![wordoftheday](https://i.ibb.co/bjqC74HM/wordoftheday.png)

The [@wordoftheday.bsky.social](https://bsky.app/profile/wordoftheday.bsky.social) bot has been running for over a year, mostly without any issues. A couple times I had to update the regex pattern to accommodate unseen definition tags but other than that it's been running smoothly and gets a moderate amount of engagement (likes, reposts, etc). You can find the source code [here](https://github.com/enerrio/dictionary-bot).

It was a fun, short project that helped me build upon my AWS skills and also learn more about Bluesky's developer ecosystem. Recently I decided to make another bot and use some new tools and techniques I've picked up since then.

## Triplets
I started thinking about ideas for another Bluesky bot and, with recent tumult in the stock market, decided to do a financial market tracker. After some more brainstorming, this turned into three bots that would track the financial system: domestic US stock market, international stock indexes, and the futures market. Each bot would post every half hour during open market hours what the current price (and daily percentage change) is for a few key tickers. Like the last project I would need to find a source to collect the data from. Thankfully there are a lot of resources for pulling financial data. So I went with [yfinance](https://yfinance-python.org) which sources ticker data from Yahoo Finance. From there, the project kinda took a similar path to the previous bot but with some improvements.

First, I used [uv](https://astral.sh/blog/uv) to manage my python dependencies which helped a lot with development. I am used to using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) for managing my Python environments but sharing these environments can be a little flimsy, especially when you need to share across platforms like CentOS to Windows. Usually you export your conda environment to a yaml file and then commit that file to Git. I've found uv to make things much easier for keeping everything standardized and it also comes with Ruff, a fast formatter, built-in.

Second, I paid a little more attention to the file structure of my project to keep things organized. Although I had three bots in one repo, I kept all bot-specific logic in separate folders. I had a `domestic` folder, an `international` folder, and a `futures` folder for each bot. Then there was a `common` folder for shared functionality like getting stock data, formatting messages, and connecting to the Bluesky account. This structure made everything much more clear to me and reduced the risk of changes to one bot affecting another.

Finally, I used a Makefile to create shortcuts for common commands I needed to run. I didn't realize that Makefiles could be used for Python projects until reading this [article](https://earthly.dev/blog/python-makefile/). The most useful shortcut was one that created the zip files that would be uploaded to AWS. Each bot requires its own zip file that packages the source code and dependencies together. In my word of the day repo I had documented in the README the instructions for how to create the zip file. There were many commands to run and it was tedious. This time, I moved all that into a shell script, generalized it to create three zip files, and added a shortcut to the Makefile to kick off that script. Now I only have to run 2 commands to update my Lambda function: `make package` and the AWS CLI command to upload that zip file to my existing Lambda function (this one I had written down in my personal notes so I wouldn't have to commit it to Git).

Although these new bots are running much more frequently than the word of the day bot, they still fall under the free tier of AWS.

![domestic](https://i.ibb.co/k2ZV4mCW/tickrbot.png)

![international](https://i.ibb.co/PZZnKtqr/tickerintlbot.png)

![futures](https://i.ibb.co/8nLQzyXL/tickrfuturebot.png)

[@tickrbot.bsky.social](https://bsky.app/profile/tickrbot.bsky.social), [@tickrbotintl.bsky.social](https://bsky.app/profile/tickrbotintl.bsky.social), and [@tickrbotfutures.bsky.social](https://bsky.app/profile/tickrbotfutures.bsky.social) have been running reliably for a couple weeks now and hopefully will not need a lot of maintenance going forward. And the source code for all three bots is in my [stock-bots repo](https://github.com/enerrio/stock-bots/tree/main).


# Lessons in building bots (and other projects)

Through these projects, I deepened my understanding of event-driven architectures, improved my skills in regex and JSON parsing, and became comfortable deploying automated workflows on AWS. In addition to that, I think there are three high level ideas I improved on and definitely recommend others to think about when starting their own projects:

* **Reliability**: Thoroughly test your code on diverse input data, including edge cases. In the word of the day bot, it was crucial to test the definition-cleaning regex pattern on a wide range of words to make sure it was working as expected.
* **Cost Management**: Always estimate operating costs and see if free tiers or trials cover your needs.
* **Organization**: Adopt widely-used tools and best practices. Structure projects clearly (pro tip: ask an LLM for feedback on your project structure. I fed it the output of the [tree](https://formulae.brew.sh/formula/tree) command). And document meticulously for your future self (use a `README.md` file for the public and a `Notes.md` for private use).

## Future of bots

Social media bots, like much technology, can be a double-edged sword. Sometimes bad actors can use them for nefarious purposes and other times they can be harmless or even provide value to a community. As long as social media platforms exist, bots will continue to evolve. Engineers creating their own bots should aim to add value ethically and responsibly. Social media is always changing so I don't expect these bots to exist forever. The data source could go away, the platform's policies could change, or you could forget your AWS login 😉


## Links
[^1]: Social media crypto scams: https://viterbischool.usc.edu/news/2022/07/usc-isi-researchers-track-crypto-pump-and-dump-operations-on-social-media/
[^2]: Earthquake bot: https://eqbot.com
[^3]: Adopt a pet bot: https://bsky.app/profile/did:plc:huey5xufsv67u3fmmtatj2ox
[^4]: Word of the day bot: https://bsky.app/profile/wordoftheday.bsky.social
[^5]: Bluesky bot doc: https://docs.bsky.app/docs/starter-templates/bots
[^6]: Merriam Webster API: https://dictionaryapi.com
