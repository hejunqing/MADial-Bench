from tqdm import tqdm
import json
import numpy as np
# AP
def average_precision_at_k(ground_truth, retrieval_results, k):
    if not ground_truth:
        return 0.0
    retrieval_results = retrieval_results[:k]
    score = 0.0
    num_hits = 0.0
    for i, result in enumerate(retrieval_results):
        if result in ground_truth and result not in retrieval_results[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(ground_truth), k)

# MRR
def reciprocal_rank_at_k(actual, predicted, k):
    mrr=0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            return 1.0 / (i + 1.0)
    return mrr

# nDCG
def dcg_at_k(relevance_scores, k):
    relevance_scores = relevance_scores[:k]
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
def ndcg_at_k(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]

    relevance_scores = [1 if p in actual else 0 for p in predicted]
    idcg = dcg_at_k(sorted(relevance_scores, reverse=True), k)
    if not idcg:
        return 0.0
    return dcg_at_k(relevance_scores, k) / idcg

# Recall
def recall_at_k(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(predicted) & set(actual)) / len(actual)

# Precision
def precision_at_k(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(predicted) & set(actual)) / k

# k_list = [1, 3, 5, 10]
# # res_path_list = ['jina', 'acge', 'stella', 'bge', 'bge_m3', 'dmeta']
# # res_path_list = ['jina', 'bge_m3', 'gte']
# res_path_list = ['openai']
#
# for model in res_path_list:
#     res_1_3_5_10 = [[], [], [], []]
#     # res_path = 'Mem-bench/en/' + model + '_top_20.json'
#     # res_path = 'Mem-bench/en/openai_top_20_bottom.json'
#     res_path = 'Mem-bench/openai_top_20_bottom.json'
#     # res_path = 'Mem-bench/en/openai_top_20.json'
#     count = 0
#     with open(res_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             count += 1
#     with open(res_path, 'r', encoding='utf-8') as file:
#         for line in tqdm(file, total=count, ncols=66):
#             sample = json.loads(line)
#             ref = sample['relevant-id']
#             res = sample['top-20-ids']
#             for i in range(len(k_list)):
#                 res_1_3_5_10[i].append(ndcg_at_k(ref, res, k_list[i]))
#     print(model)
#     print(sum(res_1_3_5_10[0]) / len(res_1_3_5_10[0]))
#     print(sum(res_1_3_5_10[1]) / len(res_1_3_5_10[1]))
#     print(sum(res_1_3_5_10[2]) / len(res_1_3_5_10[2]))
#     print(sum(res_1_3_5_10[3]) / len(res_1_3_5_10[3]))
#
#     print(round(sum(res_1_3_5_10[0]) / len(res_1_3_5_10[0]) * 100, 2))
#     print(round(sum(res_1_3_5_10[1]) / len(res_1_3_5_10[1]) * 100, 2))
#     print(round(sum(res_1_3_5_10[2]) / len(res_1_3_5_10[2]) * 100, 2))
#     print(round(sum(res_1_3_5_10[3]) / len(res_1_3_5_10[3]) * 100, 2))
#     print('\n')

import numpy as np
def geometric_average(ndcg, mrr, map, recall, precision):
    return np.exp(np.mean(np.log([ndcg, mrr, map, recall, precision])))

k_list = [1, 3, 5, 10]
res_path_list = ['BGE_M3_colbert', 'Acge', 'Stella', 'BGE_M3_dense', 'openai', 'Dmeta']
# res_path_list = ['jina', 'bge_m3', 'gte', 'openai']
# res_path_list = ['openai']

for model in res_path_list:
    map_res_1_3_5_10 = [[], [], [], []]
    mrr_res_1_3_5_10 = [[], [], [], []]
    ndcg_res_1_3_5_10 = [[], [], [], []]
    recall_res_1_3_5_10 = [[], [], [], []]
    precision_res_1_3_5_10 = [[], [], [], []]
    res_path = 'embeddings/' + model + '_zh_top_20_ids.json'
    # res_path = 'Mem-bench/en/openai_top_20_bottom.json'
    # res_path = 'Mem-bench/en/openai_top_20_bottom.json'
    # res_path = 'Mem-bench/en/openai_top_20.json'
    count = 0
    with open(res_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
    with open(res_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=count, ncols=66):
            sample = json.loads(line)
            ref = sample['relevant_id']
            res = sample['top_20_ids']
            for i in range(len(k_list)):
                map_res_1_3_5_10[i].append(average_precision_at_k(ref, res, k_list[i]))
                mrr_res_1_3_5_10[i].append(reciprocal_rank_at_k(ref, res, k_list[i]))
                ndcg_res_1_3_5_10[i].append(ndcg_at_k(ref, res, k_list[i]))
                recall_res_1_3_5_10[i].append(recall_at_k(ref, res, k_list[i]))
                precision_res_1_3_5_10[i].append(precision_at_k(ref, res, k_list[i]))
    print(model)
    avg_map = [[] for i in range(4)]
    avg_mrr = [[] for i in range(4)]
    avg_ndcg = [[] for i in range(4)]
    avg_recall = [[] for i in range(4)]
    avg_precision = [[] for i in range(4)]
    for i in range(4):
        avg_map[i].append(sum(map_res_1_3_5_10[i]) / len(map_res_1_3_5_10[i]))
        avg_mrr[i].append(sum(mrr_res_1_3_5_10[i]) / len(mrr_res_1_3_5_10[i]))
        avg_ndcg[i].append(sum(ndcg_res_1_3_5_10[i]) / len(ndcg_res_1_3_5_10[i]))
        avg_recall[i].append(sum(recall_res_1_3_5_10[i]) / len(recall_res_1_3_5_10[i]))
        avg_precision[i].append(sum(precision_res_1_3_5_10[i]) / len(precision_res_1_3_5_10[i]))
    print('map:',avg_map)
    print('mrr:',avg_mrr)
    print('ndcg',avg_ndcg)
    print('recall',avg_recall)
    print('precision',avg_precision)
    for i in range(4):
        one = geometric_average(avg_map[i][0], avg_mrr[i][0], avg_ndcg[i][0], avg_recall[i][0], avg_precision[i][0])
        print('@k: ', round(one * 100, 2))


    print('\n')

'''
BGE_M3_colbert
map: [[0.5125], [0.41892361111111115], [0.4375937500000001], [0.47487615740740763]]
mrr: [[0.5125], [0.6041666666666666], [0.6207291666666667], [0.6331374007936508]]
ndcg [[0.5125], [0.629465636734198], [0.6485236043346825], [0.6678229568441749]]
recall [[0.2798511904761905], [0.45720238095238097], [0.5714285714285714], [0.715654761904762]]
precision [[0.5125], [0.3145833333333336], [0.2487500000000001], [0.16375000000000017]]
@k:  45.41
@k:  46.99
@k:  47.83
@k:  47.24


100%|████████████████████████| 160/160 [00:00<00:00, 13178.24it/s]
Acge
map: [[0.525], [0.46267361111111105], [0.46804687500000003], [0.5089797867063492]]
mrr: [[0.525], [0.6333333333333333], [0.6495833333333334], [0.6616964285714287]]
ndcg [[0.525], [0.6668882099753899], [0.6913389547552465], [0.7029511819500122]]
recall [[0.30693452380952385], [0.5025148809523808], [0.5898214285714285], [0.735297619047619]]
precision [[0.525], [0.3395833333333336], [0.2500000000000002], [0.16812500000000025]]
@k:  47.16
@k:  50.65
@k:  49.92
@k:  49.35


100%|████████████████████████| 160/160 [00:00<00:00, 13172.81it/s]
Stella
map: [[0.53125], [0.44965277777777796], [0.45928298611111124], [0.5001395089285714]]
mrr: [[0.53125], [0.6354166666666667], [0.6554166666666668], [0.6664533730158733]]
ndcg [[0.53125], [0.6651035117141696], [0.6968798163951162], [0.7037293701167899]]
recall [[0.29995535714285715], [0.4887648809523809], [0.592842261904762], [0.7396130952380953]]
precision [[0.53125], [0.329166666666667], [0.25000000000000006], [0.16812500000000025]]
@k:  47.39
@k:  49.78
@k:  49.95
@k:  49.32


100%|████████████████████████| 160/160 [00:00<00:00, 13256.60it/s]
BGE_M3_dense
map: [[0.525], [0.44513888888888886], [0.4554479166666668], [0.4881575727513228]]
mrr: [[0.525], [0.6354166666666665], [0.6494791666666665], [0.6579365079365079]]
ndcg [[0.525], [0.6661103904378367], [0.6861068196950738], [0.6929610192824083]]
recall [[0.28943452380952384], [0.5009523809523809], [0.5856994047619046], [0.7144047619047618]]
precision [[0.525], [0.33750000000000036], [0.2525000000000001], [0.16187500000000013]]
@k:  46.61
@k:  50.19
@k:  49.6
@k:  48.1


100%|████████████████████████| 160/160 [00:00<00:00, 12931.66it/s]
openai
map: [[0.64375], [0.5472222222222223], [0.5597534722222222], [0.5981542658730159]]
mrr: [[0.64375], [0.7343749999999999], [0.7459374999999999], [0.7542832341269842]]
ndcg [[0.64375], [0.7589881195206515], [0.7689688197501329], [0.7718768629078077]]
recall [[0.36276785714285714], [0.584345238095238], [0.6943005952380952], [0.8236904761904764]]
precision [[0.64375], [0.3979166666666669], [0.2962500000000003], [0.18812500000000026]]
@k:  57.4
@k:  58.91
@k:  58.07
@k:  55.77


100%|████████████████████████| 160/160 [00:00<00:00, 13153.18it/s]
Dmeta
map: [[0.5], [0.45746527777777785], [0.4663784722222223], [0.49210662910997743]]
mrr: [[0.5], [0.6239583333333333], [0.6427083333333333], [0.6506919642857143]]
ndcg [[0.5], [0.6629820018498171], [0.6912427780665035], [0.703125167763156]]
recall [[0.3052678571428572], [0.5206994047619047], [0.5986755952380953], [0.7007142857142857]]
precision [[0.5], [0.33125000000000043], [0.24875000000000003], [0.15312500000000023]]
@k:  45.3
@k:  50.44
@k:  49.87
@k:  47.49
'''