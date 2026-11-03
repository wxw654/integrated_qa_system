import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Config:
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_PASSWORD = '1234'
    REDIS_DB = 0


if __name__ == '__main__':
    # print(Config.Redis_HOST)
    conf = Config()
    print(conf.REDIS_HOST)















