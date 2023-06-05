import argparse
import time
import mindspore
from loguru import logger
from datetime import datetime
import os
from os.path import join, exists
import pickle
from utils_CPM import set_random_seed
from mindspore import ops, nn, ms_function
from mindspore.dataset import GeneratorDataset
from mindnlp.models import GPT2LMHeadModel, GPT2Config
from mindnlp.modules import Accumulator
from dataset import CPMDataset
from mindnlp.abc.models import PreTrainedModel
from mindnlp._legacy.amp import DynamicLossScaler, all_finite

# 设置使用哪些显卡进行训练
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='2', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--vocab_path', default='CPM-mindspore/vocab/chinese_vocab.model', type=str, required=False,
                        help='sp模型路径')
    parser.add_argument('--model_config', default='CPM-mindspore/config/cpm-medium.json', type=str, required=False,
                        help='需要从头训练一个模型时，模型参数的配置文件')
    parser.add_argument('--train_path', default='CPM-mindspore/data/train.pkl', type=str, required=False, help='经过预处理之后的数据存放路径')
    parser.add_argument('--max_len', default=256, type=int, required=False, help='训练时，输入数据的最大长度')

    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='对于ignore_index的label token不计算梯度')
    parser.add_argument('--epochs', default=40, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=14, type=int, required=False, help='训练的batch size')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation_steps', default=6, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--output_path', default='CPM-mindspore/output/train', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False,
                        help='预训练的模型的路径')
    parser.add_argument('--seed', type=int, default=1234, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up步数')
    args = parser.parse_args()
    return args


def load_dataset(logger, args):
    """
    加载训练集
    """
    logger.info("loading training dataset")
    train_path = args.train_path

    with open(train_path, "rb") as f:
        train_list = pickle.load(f)

    # test
    # train_list = train_list[:24]
    logger.info('len of train data:{}'.format(len(train_list)))
    train_dataset = CPMDataset(train_list, args.max_len)

    return train_dataset


def train_epoch(train_dataloader, logger,
                epoch, args, model, train_step):
    epoch_start_time = datetime.now()
    
    total_loss = 0  # 记录下整个epoch的loss的总和
    epoch_correct_num = 0   # 每个epoch中,预测正确的word的数量
    epoch_total_num = 0  # 每个epoch中,预测的word的总数量
    
    # step begin
    model.set_train()
    for batch_idx, (input_ids, labels) in enumerate(train_dataloader.create_tuple_iterator()):
        # 捕获cuda out of memory exception
        try:
            labels=labels.astype(mindspore.int32)
            outputs = model(input_ids, labels=labels)
            logits = outputs[1]
            
            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=args.ignore_index)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            loss = train_step(input_ids, labels)
            total_loss += (loss.asnumpy().item(0) * args.gradient_accumulation_steps)
            
            if (batch_idx + 1) % args.log_step == 0:
                step = epoch * len(train_dataloader) + batch_idx
                logger.info(
                    "batch {} of epoch {}, step {}, loss {}, batch_acc {}".format(
                        batch_idx + 1, epoch + 1, step, loss.asnumpy().item(0) * args.gradient_accumulation_steps, batch_acc))
                
            del input_ids, outputs
            
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
            else:
                logger.info(str(exception))
                raise exception
            
    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))
    
     # save model
    logger.info('saving model for epoch {}'.format(epoch + 1))
    model_path = join(args.output_path, 'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    PreTrainedModel.save(model_to_save, model_path)
    # model_path = f"{model_path}/mindspore_model.ckpt"
    # save_checkpoint(model_to_save, model_path)
    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    
    return epoch_mean_loss


def train(logger, train_dataloader, args, model):

    accumulate_step = args.gradient_accumulation_steps
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=args.lr, eps=args.eps, weight_decay=1e-2)
    accumulator = Accumulator(optimizer, accumulate_step)
    
    # loss_scaler = DynamicLossScaler(scale_value=2**10, scale_factor=2, scale_window=50)
    
    def forward_fn(data, label):
        logits = model(data, labels=label)
        loss = logits[0]
        return loss / accumulate_step

    # Get gradient function
    grad_fn = mindspore.value_and_grad(forward_fn, None, model.trainable_params())

    # Define function of one-step training
    @mindspore.jit
    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        loss = ops.depend(loss, accumulator(grads))
        # loss = loss_scaler.unscale(loss)
        # status = all_finite(grads)
        # if status:
            # grads = loss_scaler.unscale(grads)
            # loss = ops.depend(loss, accumulator(grads))
        # loss = ops.depend(loss, loss_scaler.adjust(status))
        return loss

    logger.info('start training')

    train_losses = []   # 记录每个epoch的平均loss
    # ========== start training ========== #
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            train_dataloader=train_dataloader,
            logger=logger, epoch=epoch, args=args, 
            model=model, train_step=train_step)
        train_losses.append(round(train_loss, 4))
        logger.info("train loss list:{}".format(train_losses))

    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    
            
def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].view(-1, logit.shape[-1])
    labels = labels[..., 1:].view(-1)

    logit = logit.argmax(axis=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.equal(labels).masked_select(non_pad_mask).sum()
    n_correct = n_correct.asnumpy().item(0)
    n_word = non_pad_mask.sum().asnumpy().item(0)
    return n_correct, n_word


def main():
    
    args = set_args()
    
    args.cuda = not args.no_cuda

    # 创建日志对象
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info('using device:{}'.format(args.device))

    # 设置随机种子
    set_random_seed(args.seed)

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataset = load_dataset(logger, args)

    train_dataloader = GeneratorDataset(train_dataset, column_names=["input_ids", "labels"], 
                                        num_parallel_workers=args.num_workers, shuffle=True,
                                        )
    train_dataloader = train_dataloader.padded_batch(batch_size=args.batch_size, drop_remainder=True, 
                                                     pad_info={"input_ids":([args.max_len],5), "labels":([args.max_len],-100)})
    train_dataloader = train_dataloader.shuffle(20)
    
     # 创建模型
    if args.pretrained_model:  # 加载预训练模型
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:  # 初始化模型
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config, ignore_index=args.ignore_index)
        
    # 创建模型的输出目录
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    logger.info('model config:\n{}'.format(model.config.to_json_string()))
        
    # 计算模型参数数量
    num_parameters = 0
    parameters = model.trainable_params()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info("Number of teacher parameter: %.2fM" % (num_parameters / 1e6))

    # 记录参数设置
    logger.info("args:{}".format(args))
    
    train(logger, train_dataloader, args, model)


if __name__ == '__main__':
    main()

