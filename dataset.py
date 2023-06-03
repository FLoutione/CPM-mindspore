import mindspore


class CPMDataset:
    """_summary_
    """
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        # 不确定mindspore.Tensor能不能直接包list
        input_ids = mindspore.Tensor(input_ids).astype(mindspore.int64)
        return input_ids, input_ids

    def __len__(self):
        return len(self.input_list)