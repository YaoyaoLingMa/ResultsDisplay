# This is a sample Python script.
import json
import os.path
import time
import praw
import pandas as pd
from tqdm import tqdm
import datetime as dt
from datetime import timedelta
import requests
from tqdm import tqdm

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def log_action(action):
    print(action)
    return

subreddits = ['Art']
def scrape_posts(data_dir: str,start_year,end_year,api):
    # directory on which to store the data
    basecorpus = data_dir
    LOG_EVERY = 100
    LIMIT = None

    ### BLOCK 1 ###
    for year in range(start_year, end_year + 1):
        action = "[Year] " + str(year)
        log_action(action)

        dirpath = basecorpus + str(year)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # timestamps that define window of posts
        ts_after = int(dt.datetime(year, 1, 1).timestamp())
        ts_before = int(dt.datetime(year + 1, 1, 1).timestamp())

        ### BLOCK 2 ###
        for subreddit in subreddits:
            start_time = time.time()

            action = "\t[Subreddit] " + subreddit
            log_action(action)

            subredditdirpath = dirpath + '/' + subreddit
            if os.path.exists(subredditdirpath):
                pass
            else:
                os.makedirs(subredditdirpath)

            submissions_csv_path = str(year) + '-' + subreddit + '-submissions.csv'

            ### BLOCK 3 ###

            submissions_dict = {
                "id": [],
                "url": [],
                "title": [],
                "score": [],
                "num_comments": [],
                "created_utc": [],
                "selftext": [],
                "author_id": [],
                "upvote_ratio": [],
                "ups": [],
                "downs": [],
                "gilded": [],
                "top_awarded_type": [],
                "total_awards_received": [],
                "all_awardings": [],
                "awarders": [],
                "approved_at_utc": [],
                "num_reports": [],
                "removed_by": [],
                "view_count": [],
                "preview": [],
                "num_crossposts": [],
                "link_flair_text": [],
                "whitelist_status": []

            }

            ### BLOCK 4 ###

            # use PSAW only to get id of submissions in time interval
            gen = api.search_submissions(
                after=ts_after,
                before=ts_before,
                # filter=['id'],
                subreddit=subreddit,
                limit=LIMIT
            )

            #print(api.metadata_)

            ### BLOCK 5 ###

            # use PMAW to get actual info and traverse comment tree
            for idx, submission_pmaw in enumerate(gen):

                submission_id = submission_pmaw['id']

                submissions_dict["id"].append(submission_pmaw['id'])
                submissions_dict["url"].append(submission_pmaw['url'])
                submissions_dict["title"].append(submission_pmaw['title'])
                submissions_dict["score"].append(submission_pmaw['score'])
                submissions_dict["num_comments"].append(submission_pmaw['num_comments'])
                submissions_dict["created_utc"].append(submission_pmaw['created_utc'])
                submissions_dict["selftext"].append(submission_pmaw['selftext'])
                submissions_dict["upvote_ratio"].append(submission_pmaw['upvote_ratio'])
                submissions_dict["ups"].append(submission_pmaw['ups'])
                submissions_dict["downs"].append(submission_pmaw['downs'])
                submissions_dict["gilded"].append(submission_pmaw['gilded'])
                submissions_dict["top_awarded_type"].append(submission_pmaw['top_awarded_type'])
                submissions_dict["total_awards_received"].append(submission_pmaw['total_awards_received'])
                submissions_dict["all_awardings"].append(submission_pmaw['all_awardings'])
                submissions_dict["awarders"].append(submission_pmaw['awarders'])
                submissions_dict["approved_at_utc"].append(submission_pmaw['approved_at_utc'])
                submissions_dict["num_reports"].append(submission_pmaw['num_reports'])
                submissions_dict["removed_by"].append(submission_pmaw['removed_by'])
                submissions_dict["view_count"].append(submission_pmaw['view_count'])
                submissions_dict["num_crossposts"].append(submission_pmaw['num_crossposts'])
                submissions_dict["link_flair_text"].append(submission_pmaw['link_flair_text'])
                submissions_dict["whitelist_status"].append(submission_pmaw['whitelist_status'])

                try:
                    submissions_dict["preview"].append(submission_pmaw['preview'])
                except:
                    submissions_dict["preview"].append(-1)
                try:
                    submissions_dict["author_id"].append(submission_pmaw['author'].id)
                except:
                    submissions_dict["author_id"].append(-1)

                if idx % LOG_EVERY == 0 and idx > 0:
                    time_delta = time.time() - start_time
                    action = f"\t\t[Info] {idx} submissions processed after {timedelta(seconds=time_delta)}"
                    log_action(action)

            ### BLOCK 6 ###

            # single csv file with all submissions
            pd.DataFrame(submissions_dict).to_csv(subredditdirpath + '/' + submissions_csv_path,
                                                  index=False)

            action = f"\t\t[Info] Found submissions: {pd.DataFrame(submissions_dict).shape[0]}"
            log_action(action)

            action = f"\t\t[Info] Elapsed time: {time.time() - start_time: .2f}s"
            log_action(action)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    CLIENT_ID = 'E5uJPpvcfwHt8daLuguBQw'
    CLIENT_SECRET = '59aJe1uHA5XchkIw7UEkdtAI3g3cjA'
    USER_AGENT = 'myagent'
    subreddit_str = 'Art'

    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT,
                         password="lky.reddit.pwd",
                         username="yaoyao0ma", ratelimit_seconds=600)
    #api = PushshiftAPI(praw=reddit)
    #get_submission_ids(reddit, 2022, 2023)
    subreddit = reddit.subreddit(subreddit_str)

    #scrape_posts('D:/日常工作学习文档/数字媒体导论/ArtImageFromReddit/', 2020, 2023, api)
    print('Starting scrap image from r/Art,target number = 1000')
    correct_cnt = 0
    for submission in tqdm(subreddit.hot(limit=2000)):

        imagespath = '/data/liukai/likeyao/images/ArtFromReddit/images/'
        submission_image_path = f"{submission.id}-image"
        submission_images_path = os.path.join(imagespath, submission_image_path)
        if os.path.exists(submission_images_path+'.jpeg') or os.path.exists(submission_images_path+'.png'):
            correct_cnt += 1
            if correct_cnt == 1000: break
        else:
            try:
                response = requests.get(submission.url, stream=True)
            except Exception as first_e:
                # print("Couldn't get source image from preview")
                # print(e)
                try:
                    response = requests.get(submission.url, stream=True)
                except Exception as e:
                    correct_cnt -= 1
                    if correct_cnt<0: correct_cnt = 0
                    print("Couldn't get source image from either preview or url")
                    print("First E", first_e)
                    print(e)
                    continue
            if not response.ok:
                print(response)
                correct_cnt -= 1
                if correct_cnt < 0: correct_cnt = 0
                continue

            try:
                _, extension = response.headers["content-type"].split("/")
            except Exception as e:
                print(e)
                correct_cnt -= 1
                if correct_cnt < 0: correct_cnt = 0
                continue

            try:
                with open(f'{submission_images_path}.{extension}', 'wb') as handle:
                    for block in response.iter_content(1024):
                        if not block:
                            break
                        handle.write(block)
                correct_cnt += 1
                if correct_cnt == 1000: break
            except Exception as e:
                print(e)
                correct_cnt -= 1
                if correct_cnt < 0: correct_cnt = 0
                continue
    print('Art Image Scrap Done! Number:'+str(correct_cnt))


