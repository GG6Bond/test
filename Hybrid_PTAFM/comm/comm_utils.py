from .nccl_backend import *

_DATA_PARALLEL_COMM = None
_DATA_PARALLEL_RANK = None
_DATA_PARALLEL_WORLD_SIZE = None

_PIPELINE_PARALLEL_COMM = None
_PIPELINE_PARALLEL_RANK = None
_PIPELINE_PARALLEL_WORLD_SIZE = None

# PTAFM
_PIPELINE_PARALLEL_SCATTER_COMM =None
_PIPELINE_PARALLEL_GATHER_COMM =None

_GATHE_WORLD_SIZE = None
_SCATTER_WORLD_SIZE = None

# 0-T4  1-A100
_DEVICE_GPU = None

_PIPELINE_GATHER_RANK =None
_PIPELINE_SCATTER_RANK =None

rank_map=[12, 13, 6, 2, 3, 0, 1, 7, 8, 9, 10, 11, 4, 5, 18, 19, 20, 21, 22, 23, 42, 43, 36, 37, 38, 39, 40, 41, 26, 27, 28, 29, 34, 35, 44, 45, 46, 47, 24, 25, 30, 31, 32, 33, 14, 15, 16,17]

rank_A100 = [12,13,2,3,4,5,6,7,8,9,10,11]

rank_mapping_id=[36, 37, 6, 9, 12, 15, 18, 21, 24, 27, 30 ,33, 0, 3, 38, 39, 40, 41, 42, 43, 44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65, 66, 67, 68, 69, 70, 71 ]


# 数据并行组、流水线并行组、gather和scatter操作的通信组
data_parallel_config = [[9,12,51,50,43,52,46,44],[65,63,62,64,61,60,0,3],[6,53,15,48,47,49,42,45],
                        [33,30,57,55,58,56,54,59],[21,24,27,18],
                        [66,71,69,70,67,68,38,40,37,36,39,41]]

pipeline_config = [[66,71,69,21,33,6,65,63,62,9],[70,67,68,24,30,15,64,61,60,12],
                   [38,40,37,27,57,55,58,53,48,47,0,51,50,43],[36,39,41,18,56,54,59,49,42,45,3,52,46,44]
]

scatters_comm = [[6,65,63,62],[15,64,61,60],[0,51,50,43],[27,57,55,58],[3,52,46,44],[18,56,54,59]]
# 流水线并行小的通讯组scatter:第一位是scatter的首发地，最后一位是gather的目的地
gathers_comm = [[65,63,62,9], [66,71,69,21],[64,61,60,12],[70,67,68,24],[53,48,47,0],[38,40,37,27],[49,42,45,3],[36,39,41,18]]


def find_list(id, config):
    for i, sublist in enumerate(config):
        if id in sublist:
            return i, len(sublist)
    return None, None

def get_device_gpu() -> int:
    return _DEVICE_GPU

# 返回当前节点在gather操作中的排名
def get_pipeline_gather_rank() -> int:
    assert _PIPELINE_GATHER_RANK is not None
    return _PIPELINE_GATHER_RANK

# 当前节点在scatter操作中的排名
def get_pipeline_scatter_rank() -> int:
    assert _PIPELINE_SCATTER_RANK is not None
    return _PIPELINE_SCATTER_RANK

# 返回gather、scatter通信组的大小
def get_gather_world_size() -> int:
    assert _GATHE_WORLD_SIZE is not None
    return _GATHE_WORLD_SIZE
def get_scatter_world_size() -> int:
    assert _SCATTER_WORLD_SIZE is not None
    return _SCATTER_WORLD_SIZE

# 返回gather和scatter操作的通信器
def get_pipeline_gather_comm() -> NCCLCommunicator:
    #assert _PIPELINE_PARALLEL_GATHER_COMM is not None
    return _PIPELINE_PARALLEL_GATHER_COMM

def get_pipeline_scatter_comm() -> NCCLCommunicator:
    #assert _PIPELINE_PARALLEL_SCATTER_COMM is not None
    return _PIPELINE_PARALLEL_SCATTER_COMM


def get_data_parallel_comm() -> NCCLCommunicator:
    assert _DATA_PARALLEL_COMM is not None
    return _DATA_PARALLEL_COMM


def get_data_parallel_rank() -> int:
    assert _DATA_PARALLEL_RANK is not None
    return _DATA_PARALLEL_RANK


def get_data_parallel_world_size() -> int:
    assert _DATA_PARALLEL_WORLD_SIZE is not None
    return _DATA_PARALLEL_WORLD_SIZE


def get_pipeline_parallel_comm() -> NCCLCommunicator:
    assert _PIPELINE_PARALLEL_COMM is not None
    return _PIPELINE_PARALLEL_COMM


def get_pipeline_parallel_rank() -> int:
    assert _PIPELINE_PARALLEL_RANK is not None
    return _PIPELINE_PARALLEL_RANK


def get_pipeline_parallel_world_size() -> int:
    assert _PIPELINE_PARALLEL_WORLD_SIZE is not None
    return _PIPELINE_PARALLEL_WORLD_SIZE


def init_communicators(args):
    #print(str(_DEVICE_GPU.shape))
    default_init(args)
    #print("1")
    #real_node_count = args.world_size+1
    #if args.rank==2:

    #args.world_size=args.world_size+1
    #assert args.world_size == args.data_group_size * args.pipeline_group_size
    #print("2") 
    print(f"{args.world_size} == {args.data_group_size} * {args.pipeline_group_size}")
    if args.data_group_size != args.data_group_size * args.pipeline_group_size:
        #    We do the following hard code alignment of communication groups:
        #    Suppose there are 8 instances (world_size), and 4 data parallel groups (data_group_size is 2),
        #    Then there would be 2 pipeline parallel groups (pipeline_group_size is 4), then the groups will look like:
        #    pipeline parallel: <group 0: [0,1,2,3]>, <group 1: [4,5,6,7]>
        #    data parallel: <group 0: [0,4]>, <group 1: [1,5]>, <group 2: [2,6]>, <group 3: [3,7]>
        #assert args.world_size == args.data_group_size * args.pipeline_group_size
        global _DATA_PARALLEL_COMM
        global _PIPELINE_PARALLEL_COMM
        global _DATA_PARALLEL_RANK
        global _PIPELINE_PARALLEL_RANK
        global _DATA_PARALLEL_WORLD_SIZE
        global _PIPELINE_PARALLEL_WORLD_SIZE
       
        # gather & scatter

        global _GATHE_WORLD_SIZE
        global _SCATTER_WORLD_SIZE
        global _DEVICE_GPU

        global _PIPELINE_GATHER_RANK
        global _PIPELINE_SCATTER_RANK
    #    print(str(_DEVICE_GPU.shape))
        mapping_rank = rank_mapping_id[args.rank]
        if args.rank in rank_A100:
            _DEVICE_GPU = 1
            print("A100 Node"+str(args.rank)+". MappingID:"+str(mapping_rank))
        else:
            print("T4 Node"+str(args.rank)+". MappingID:"+str(mapping_rank))
            _DEVICE_GPU=0
        # We use pipeline parallel by default.
        _PIPELINE_PARALLEL_WORLD_SIZE = args.pipeline_group_size
        #_PIPELINE_PARALLEL_RANK = args.rank % args.pipeline_group_size
        _PIPELINE_PARALLEL_RANK = args.rank
        # rank --> id mapping   21--->36
        #mapping_rank = rank_mapping_id[args.rank]
        # build pipellel group 0 11 [0,1,2,3,4,5,18,36,54,72,90]
        list_index,group_size = find_list(mapping_rank, pipeline_config)
        # [0,1,2,3,4,5,18,36,54,72,90]
        pipeline_list = pipeline_config[list_index]
        # 
        print("                                                                 rankid:"+str(_PIPELINE_PARALLEL_RANK)+",mapping id:"+str(mapping_rank)+",group :"+str(pipeline_config[list_index])+",group size:"+str(group_size))

        # id ----> rank 36----->7
        for i in range(group_size):
            if pipeline_list[i] == mapping_rank:
                _PIPELINE_PARALLEL_RANK = i
        print("_PIPELINE_PARALLEL_RANK:",_PIPELINE_PARALLEL_RANK)
        args.pipeline_group_size= group_size
        _PIPELINE_PARALLEL_WORLD_SIZE = args.pipeline_group_size

        _PIPELINE_PARALLEL_COMM = NCCLCommunicator(_PIPELINE_PARALLEL_RANK, args.cuda_id, group_size,
                                                   "pipeline_group_"+str(list_index))
        global _PIPELINE_PARALLEL_GATHER_COMM
        global _PIPELINE_PARALLEL_SCATTER_COMM


        gathercom_index, gather_group_size = find_list(mapping_rank, gathers_comm)

        if gathercom_index is not None:
            #print("")
            for i in range(gather_group_size):
                if gathers_comm[gathercom_index][i] == mapping_rank:
                    _PIPELINE_GATHER_RANK = i
                    _GATHE_WORLD_SIZE =gather_group_size
            print("Gather_id:"+str(_PIPELINE_GATHER_RANK)+",mapping id:"+str(mapping_rank)+",group :"+str(gathers_comm[gathercom_index])+",Gather group size:"+str(gather_group_size))
            _PIPELINE_PARALLEL_GATHER_COMM = NCCLCommunicator(_PIPELINE_GATHER_RANK, args.cuda_id, gather_group_size,
                                                       "pipeline_gather_group_" + str(gathercom_index))
        scattercom_index, scatter_group_size = find_list(mapping_rank, scatters_comm)
        if scattercom_index is not None:
            for i in range(scatter_group_size):
                if scatters_comm[scattercom_index][i] == mapping_rank:
                    _PIPELINE_SCATTER_RANK = i
                    _SCATTER_WORLD_SIZE = scatter_group_size
            print("Scatther_id:"+str(_PIPELINE_PARALLEL_RANK)+",mapping id:"+str(mapping_rank)+",group :"+str(scatters_comm[scattercom_index])+",Scatter group size:"+str(scatter_group_size))
            _PIPELINE_PARALLEL_SCATTER_COMM = NCCLCommunicator(_PIPELINE_SCATTER_RANK, args.cuda_id, scatter_group_size,
                                                       "pipeline_scatter_group_" + str(scattercom_index))
        
        if args.data_group_size != 1:
            list_index, data_group_size = find_list(mapping_rank, data_parallel_config)
            args.data_group_size = data_group_size
            data_para_list = data_parallel_config[list_index]
            _DATA_PARALLEL_WORLD_SIZE = args.data_group_size
            # [0,0] []
            #_DATA_PARALLEL_RANK = args.rank
            for i in range(data_group_size):
                if data_para_list[i] == mapping_rank:
                    _DATA_PARALLEL_RANK = i
            print("Data Parallel id:"+str(_DATA_PARALLEL_RANK)+",mapping id:"+str(mapping_rank)+",group :"+str(data_para_list)+",DATA group size:"+str(data_group_size))
            _DATA_PARALLEL_COMM = NCCLCommunicator(_DATA_PARALLEL_RANK, args.cuda_id, args.data_group_size,
                                                   "data_group_" + str(list_index))
            #print(str(_DATA_PARALLEL_RANK.shape))
    else:
        print("Not supported yet")
        assert False
