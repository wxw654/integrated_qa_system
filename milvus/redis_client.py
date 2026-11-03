import redis
import json
from redis_crud import Config


class RedisClient:
    def __init__(self):
        try:
            self.client = redis.StrictRedis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                password=Config.REDIS_PASSWORD,
                db=Config.REDIS_DB,
                decode_responses=True
            )
        except redis.RedisError as e:
            print("连接异常")

    def set_data(self, key, value):
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False))
            print(f"数据已成功保存到Redis:{key}")
        except Exception as e:
            print(f"e: {e}")

    def get_data(self,key):
        try:
            value=self.client.get(key)
            # return value
            return json.loads(value)  if value else None
        except Exception as e:
            print(f"e:{e}")

    def get_answer(self, query):
        try:
            answer = self.client.get(f"it培训:{query}")
            if answer:
                print(f"从redis获取答案:{query}")
                return answer
            return None

        except Exception as e:
            print(f"e:{e}")

    def delete_key(self, key):
        try:
            # redis要求同一个数据库下,key必须唯一
            result = self.client.delete(key)
            if result == 1:
                print("删除成功")
            else:
                print("删除失败")
        except Exception as e:
            print(f"e:{e}")



if __name__ == '__main__':
    redis_client = RedisClient()
    # 格式一:
    # key="黑马"
    # value={"name":"黑马","age":19}
    # 格式二:
    # key = "it培训:黑马"
    #  value = {"name": "黑马", "age": 19}
    # redis_client.set_data(key,value)
    # print(type(redis_client.get_data("黑马")))

    # print(redis_client.get_data("it培训:黑马"))

    # print(redis_client.get_answer("盒马"))
    redis_client.delete_key("it培训:黑马")

















