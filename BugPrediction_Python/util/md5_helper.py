import hashlib

class MD5_HELPER:
    def get_md5(self,content):
        result = hashlib.md5(content.encode("utf-8")).hexdigest()
        return result