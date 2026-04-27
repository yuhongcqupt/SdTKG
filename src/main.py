import argparse
import os
import sys

import dgl
import numpy as np
import torch
from tqdm import tqdm
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph,build_sub_hete_graph,get_h1
from src.rrgcn import SdTKGGCN
import torch.nn.modules.rnn
from rgcn.knowledge_graph import _read_triplets_as_list
import logging


def test(logger, model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, time_list, history_time_nogt, mode):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]


    rule_idx=[]
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)

        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph, test_triples_input, use_cuda)

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples_input, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples_input, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        with open('../output/icews14s/rel.txt', 'w', encoding='utf-8') as w_f:
            for i in ranks_filter_r:
                w_f.write(str(i))



        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1

    mrr_raw, hit_result_raw = utils.stat_ranks(logger,ranks_raw, "raw_ent")
    mrr_filter, hit_result_filter = utils.stat_ranks(logger,ranks_filter, "filter_ent")
    mrr_raw_r, hit_result_raw_r = utils.stat_ranks(logger,ranks_raw_r, "raw_rel")
    mrr_filter_r, hit_result_filter_r = utils.stat_ranks(logger,ranks_filter_r, "filter_rel")

    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r, rule_idx



def run_experiment(args,logger, history_len=None, n_layers=None, dropout=None, n_bases=None, angle=None, history_rate=None):
    # load configuration for grid search the best configuration
    if history_len:
        args.train_history_len = history_len
        args.test_history_len = history_len
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    if angle:
        args.angle = angle
    if history_rate:
        args.history_rate = history_rate
    mrr_raw = None
    mrr_filter = None
    mrr_raw_r = None
    mrr_filter_r = None
    hit_result_raw = None
    hit_result_filter = None
    hit_result_raw_r = None
    hit_result_filter_r = None

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)   # 得到data类
    train_list, train_times = utils.split_by_time(data.train)   # 划分为snapshots，逐时间步的数据集
    valid_list, valid_times = utils.split_by_time(data.valid)
    if args.test_part:
        test_list, test_times = utils.split_by_time(data.test_part)
    else:
        test_list, test_times = utils.split_by_time(data.test)
    path_list, path_times = utils.split_by_time(data.path)
    #h1_list, h1_times = utils.split_by_time(data.h1)
    path_list=np.array(path_list)
    #rule_dict = data.rule_dict
    #print(rule_dict)
    #path_dict =data.path_dict
    h1_dict = data.h1_dict

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    #num_paths = data.num_path
    """
    for i in range(num_rels):
        path_dict[str(i)] = str([i])
    """

    if args.dataset == "ICEWS14s":
        num_times = len(train_list) + len(valid_list) + len(test_list) + 1
    else:
        num_times = len(train_list) + len(valid_list) + len(test_list)
    time_interval = train_times[1]-train_times[0]
    print("num_times", num_times, "--------------", time_interval)
    history_val_time_nogt = valid_times[0]
    history_test_time_nogt = test_times[0]
    if args.multi_step:
        print("val only use global history before:", history_val_time_nogt)
        print("test only use global history before:", history_test_time_nogt)

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    model_name = "task_{}-his_{}-{}-{}-{}-ly{}-dilate{}-his{}-weight_{}-discount_{}-angle_{}-dp{}_{}_{}_{}-gpu{}-{}"\
        .format(args.task_weight,args.history_rate, args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu, args.save)


    if args.add_self_loss and not args.add_hete_graph:
        model_state_file = os.path.join('../models_w_o_hete/', model_name)
    elif args.add_hete_graph and not args.add_self_loss:
        model_state_file = os.path.join('../models_w_o_self/', model_name)
    else:
        model_state_file = os.path.join('../models/', model_name)
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
    print(model_state_file)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # create stat
    model = SdTKGGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        num_times,
                        time_interval,
                        args.n_hidden,
                        args.opn,
                        args.history_rate,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis,
                        use_hete_graph=args.add_hete_graph)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)


    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r, _ = test(logger,model,
                                                            train_list+valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test,
                                                            all_ans_list_r_test,
                                                            model_state_file,
                                                            static_graph,
                                                            test_times,
                                                            history_test_time_nogt,
                                                            "test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0


        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []


            idx = [_ for _ in range(len(train_list))]
            #random.shuffle(idx)
            for train_sample_num in tqdm(idx):
                #print(train_sample_num)
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                    #index = range(0, train_sample_num)
                    path_input_list=path_list[0:train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]
                    #index = range(train_sample_num - args.train_history_len, train_sample_num)
                    path_input_list=path_list[train_sample_num - args.train_history_len: train_sample_num]

                h1 = [np.array(get_h1(j,h1_dict)) for j in input_list]

                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                history_hete_glist = [build_sub_hete_graph(num_nodes, input_list[snap], path_input_list[snap],h1[snap],
                                         use_cuda,args.gpu) for snap in range(len(input_list))]

                loss_e, loss_r, loss_sub = model.get_loss(history_glist, output[0],  history_hete_glist, use_cuda)

                if args.add_self_loss:
                    loss = loss_r + args.task_weight * loss_sub
                else:
                    loss = loss_r

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()



                #history_glist = [build_pair_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_pair_list]
                #final_r_score = model.predict_rel(history_glist, num_rels, static_graph, input_pair_list, use_cuda)

            logger.info("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r, _ = test(logger,model,
                                                                    train_list, 
                                                                    valid_list, 
                                                                    num_rels, 
                                                                    num_nodes, 
                                                                    use_cuda, 
                                                                    all_ans_list_valid, 
                                                                    all_ans_list_r_valid, 
                                                                    model_state_file, 
                                                                    static_graph,
                                                                    valid_times,
                                                                    history_val_time_nogt,
                                                                    mode="train")

                print(args.relation_evaluation)
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:

                        best_mrr = mrr_raw
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)


        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r, _= test(logger,model,
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            static_graph,
                                                            test_times,
                                                            history_test_time_nogt,
                                                            mode="test")


    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='TIRGN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=50,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--add_hete_graph", action='store_true', default=False,
                        help="add hete graph")
    parser.add_argument("--add_self_loss", action='store_true', default=False,
                        help="add Self-supervised Learning")

    parser.add_argument("--test_part", action='store_true', default=False,
                        help="test part of data")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")



    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")


    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("-tune", "--tune", type=str, default="history_len,n_layers,dropout,n_bases,angle,history_rate",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    # configuration for global history
    parser.add_argument("--history-rate", type=float, default=0.3,
                        help="history rate")

    parser.add_argument("--save", type=str, default="one",
                        help="number of save")



    args = parser.parse_args()


    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)  # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler("train.log")
    fh.setLevel(logging.DEBUG)  # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)  # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)  # 将print函数改为logger.info函数就可以将输出信息保存到日志文件中




    run_experiment(args,logger)
    sys.exit()



