import requests
from apscheduler.scheduler.blocking import BlcokingScheduler


scheduler = BlcokingScheduler()


@scheduler.scheduled_job('interval', minutes=15)
def timed_job_awake_heroku_app():
    url = 'https://esaleshub-demo.herokuapp.com/'
    res = requests.get(url)
    print("Wake up every 15 minutes!")
    print(f'Content: \n{res.content}')


scheduler.start()