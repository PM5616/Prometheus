"""Cryptography Utility Functions

加密解密相关的工具函数。
"""

import hashlib
import hmac
import base64
import secrets
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger


class CryptoUtils:
    """加密工具类"""
    
    def __init__(self, key: Optional[bytes] = None):
        """初始化加密工具
        
        Args:
            key: 加密密钥，如果为None则生成新密钥
        """
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """加密数据
        
        Args:
            data: 要加密的数据
            
        Returns:
            str: 加密后的base64字符串
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.cipher.encrypt(data)
            return base64.b64encode(encrypted_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return ""
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据
        
        Args:
            encrypted_data: 加密的base64字符串
            
        Returns:
            str: 解密后的字符串
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return ""
    
    def get_key(self) -> str:
        """获取密钥的base64字符串
        
        Returns:
            str: 密钥的base64字符串
        """
        return base64.b64encode(self.key).decode('utf-8')
    
    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> 'CryptoUtils':
        """从密码生成加密工具
        
        Args:
            password: 密码
            salt: 盐值，如果为None则生成新盐值
            
        Returns:
            CryptoUtils: 加密工具实例
        """
        try:
            if salt is None:
                salt = secrets.token_bytes(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
            return cls(key)
        except Exception as e:
            logger.error(f"Error creating crypto utils from password: {e}")
            return cls()


def generate_api_signature(secret: str, 
                          message: str, 
                          algorithm: str = 'sha256') -> str:
    """生成API签名
    
    Args:
        secret: 密钥
        message: 要签名的消息
        algorithm: 哈希算法
        
    Returns:
        str: 签名字符串
    """
    try:
        if algorithm.lower() == 'sha256':
            hash_func = hashlib.sha256
        elif algorithm.lower() == 'sha1':
            hash_func = hashlib.sha1
        elif algorithm.lower() == 'md5':
            hash_func = hashlib.md5
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hash_func
        ).hexdigest()
        
        return signature
    except Exception as e:
        logger.error(f"Error generating API signature: {e}")
        return ""


def generate_binance_signature(secret: str, query_string: str) -> str:
    """生成币安API签名
    
    Args:
        secret: API密钥
        query_string: 查询字符串
        
    Returns:
        str: 签名字符串
    """
    try:
        signature = hmac.new(
            secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    except Exception as e:
        logger.error(f"Error generating Binance signature: {e}")
        return ""


def hash_password(password: str, salt: Optional[str] = None) -> tuple:
    """哈希密码
    
    Args:
        password: 原始密码
        salt: 盐值，如果为None则生成新盐值
        
    Returns:
        tuple: (哈希值, 盐值)
    """
    try:
        if salt is None:
            salt = secrets.token_hex(16)
        
        # 使用PBKDF2进行密码哈希
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 迭代次数
        )
        
        return password_hash.hex(), salt
    except Exception as e:
        logger.error(f"Error hashing password: {e}")
        return "", ""


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """验证密码
    
    Args:
        password: 原始密码
        password_hash: 存储的哈希值
        salt: 盐值
        
    Returns:
        bool: 验证结果
    """
    try:
        # 计算输入密码的哈希值
        computed_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        # 使用安全比较避免时序攻击
        return hmac.compare_digest(computed_hash, password_hash)
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def generate_random_token(length: int = 32) -> str:
    """生成随机令牌
    
    Args:
        length: 令牌长度
        
    Returns:
        str: 随机令牌
    """
    try:
        return secrets.token_hex(length)
    except Exception as e:
        logger.error(f"Error generating random token: {e}")
        return ""


def generate_uuid() -> str:
    """生成UUID
    
    Returns:
        str: UUID字符串
    """
    try:
        import uuid
        return str(uuid.uuid4())
    except Exception as e:
        logger.error(f"Error generating UUID: {e}")
        return ""


def encode_base64(data: Union[str, bytes]) -> str:
    """Base64编码
    
    Args:
        data: 要编码的数据
        
    Returns:
        str: Base64编码字符串
    """
    try:
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return base64.b64encode(data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding base64: {e}")
        return ""


def decode_base64(encoded_data: str) -> str:
    """Base64解码
    
    Args:
        encoded_data: Base64编码字符串
        
    Returns:
        str: 解码后的字符串
    """
    try:
        decoded_bytes = base64.b64decode(encoded_data.encode('utf-8'))
        return decoded_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Error decoding base64: {e}")
        return ""


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法
        
    Returns:
        str: 文件哈希值
    """
    try:
        if algorithm.lower() == 'sha256':
            hash_func = hashlib.sha256()
        elif algorithm.lower() == 'sha1':
            hash_func = hashlib.sha1()
        elif algorithm.lower() == 'md5':
            hash_func = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return ""


def secure_compare(a: str, b: str) -> bool:
    """安全字符串比较，防止时序攻击
    
    Args:
        a: 第一个字符串
        b: 第二个字符串
        
    Returns:
        bool: 比较结果
    """
    try:
        return hmac.compare_digest(a, b)
    except Exception as e:
        logger.error(f"Error in secure compare: {e}")
        return False


def mask_sensitive_data(data: str, 
                       mask_char: str = '*', 
                       visible_start: int = 4, 
                       visible_end: int = 4) -> str:
    """掩码敏感数据
    
    Args:
        data: 原始数据
        mask_char: 掩码字符
        visible_start: 开头可见字符数
        visible_end: 结尾可见字符数
        
    Returns:
        str: 掩码后的数据
    """
    try:
        if len(data) <= visible_start + visible_end:
            return mask_char * len(data)
        
        start = data[:visible_start]
        end = data[-visible_end:] if visible_end > 0 else ""
        middle_length = len(data) - visible_start - visible_end
        middle = mask_char * middle_length
        
        return start + middle + end
    except Exception as e:
        logger.error(f"Error masking sensitive data: {e}")
        return data


def generate_api_key_pair() -> tuple:
    """生成API密钥对
    
    Returns:
        tuple: (API Key, API Secret)
    """
    try:
        api_key = generate_random_token(32)
        api_secret = generate_random_token(64)
        return api_key, api_secret
    except Exception as e:
        logger.error(f"Error generating API key pair: {e}")
        return "", ""


def validate_api_key_format(api_key: str) -> bool:
    """验证API密钥格式
    
    Args:
        api_key: API密钥
        
    Returns:
        bool: 验证结果
    """
    try:
        # 检查长度（通常为64个字符）
        if len(api_key) != 64:
            return False
        
        # 检查是否只包含十六进制字符
        try:
            int(api_key, 16)
            return True
        except ValueError:
            return False
    except Exception as e:
        logger.error(f"Error validating API key format: {e}")
        return False


def encrypt_config_value(value: str, key: Optional[str] = None) -> str:
    """加密配置值
    
    Args:
        value: 要加密的值
        key: 加密密钥
        
    Returns:
        str: 加密后的值
    """
    try:
        if key:
            crypto = CryptoUtils(base64.b64decode(key.encode('utf-8')))
        else:
            crypto = CryptoUtils()
        
        return crypto.encrypt(value)
    except Exception as e:
        logger.error(f"Error encrypting config value: {e}")
        return value


def decrypt_config_value(encrypted_value: str, key: str) -> str:
    """解密配置值
    
    Args:
        encrypted_value: 加密的值
        key: 解密密钥
        
    Returns:
        str: 解密后的值
    """
    try:
        crypto = CryptoUtils(base64.b64decode(key.encode('utf-8')))
        return crypto.decrypt(encrypted_value)
    except Exception as e:
        logger.error(f"Error decrypting config value: {e}")
        return encrypted_value


# 为了兼容性，添加一些别名函数
def encrypt_data(data: Union[str, bytes], key: Optional[str] = None) -> str:
    """加密数据（兼容性函数）
    
    Args:
        data: 要加密的数据
        key: 加密密钥
        
    Returns:
        str: 加密后的数据
    """
    try:
        if key:
            crypto = CryptoUtils(base64.b64decode(key.encode('utf-8')))
        else:
            crypto = CryptoUtils()
        
        return crypto.encrypt(data)
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        return ""


def decrypt_data(encrypted_data: str, key: str) -> str:
    """解密数据（兼容性函数）
    
    Args:
        encrypted_data: 加密的数据
        key: 解密密钥
        
    Returns:
        str: 解密后的数据
    """
    try:
        crypto = CryptoUtils(base64.b64decode(key.encode('utf-8')))
        return crypto.decrypt(encrypted_data)
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        return ""


def hash_data(data: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """计算数据哈希值（兼容性函数）
    
    Args:
        data: 要哈希的数据
        algorithm: 哈希算法
        
    Returns:
        str: 哈希值
    """
    try:
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm.lower() == 'sha256':
            hash_func = hashlib.sha256()
        elif algorithm.lower() == 'sha1':
            hash_func = hashlib.sha1()
        elif algorithm.lower() == 'md5':
            hash_func = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_func.update(data)
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing data: {e}")
        return ""


def validate_signature(message: str, signature: str, secret: str, algorithm: str = 'sha256') -> bool:
    """验证签名（兼容性函数）
    
    Args:
        message: 原始消息
        signature: 签名
        secret: 密钥
        algorithm: 哈希算法
        
    Returns:
        bool: 验证结果
    """
    try:
        expected_signature = generate_api_signature(secret, message, algorithm)
        return secure_compare(signature, expected_signature)
    except Exception as e:
        logger.error(f"Error validating signature: {e}")
        return False