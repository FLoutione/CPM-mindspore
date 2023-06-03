import logging
import mindspore
import random
import numpy as np
from mindspore import ops
from mindspore.ops import operations as P


def set_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def set_random_seed(seed):
    """
    设置训练的随机种子
    """
    mindspore.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. 
    """
    assert len(logits.shape) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.shape[-1])  # Safety check
    logits_np = logits.asnumpy()
    logits_np = np.array(logits_np)
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # mindspore.ops.operations.Topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # mindspore.ops.TopK()操作来获取一个Tensor中每一行的前k个最大值和它们的索引
        # ...表示其他维度由计算机自行推断
        # 该注释代码是采用Tensor实现的返回最大的k个元素
        # topk = P.TopK(sorted=True)
        # less = ops.Less()
        # indices_to_remove = less(logits, topk(logits, top_k)[0][-1])
        # logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷
        
        # 采用numpy直接返回的k个最大的元素，不走Tensor，提升效率
        sorted_indices = np.argsort(logits_np)[::-1]
        indices_to_remove = sorted_indices[top_k:]
        logits_np[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits = np.sort(logits_np)[::-1]
        sorted_indices = np.argsort(logits_np)[::-1]
        
        # 对排序后的 logits 数组进行 softmax 处理
        softmax_probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
        # 计算累积概率
        cumulative_probs = np.cumsum(softmax_probs)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = np.copy(sorted_indices_to_remove[..., :-1])
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits_np = np.array(logits_np)
        logits_np[indices_to_remove] = filter_value
    return logits_np