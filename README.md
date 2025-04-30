# MADial-Bench

Repo for the paper, MADial-Bench: Towards Real-world Evaluation of Memory-Augmented Dialogue Generation. NAACL 2025.

Here we introduce a benchmark for memory-augmented chatbot by proposing two-stage memory recall and multi-recall paradigm based on cognitive science and psychological theory. 

<img width="479" alt="two-stage memory-augmented chatbot" src="https://github.com/user-attachments/assets/ff6b8fc0-5986-4cd4-8546-a9ef169facd4" />


The dialogue and memory data for testing are in **data** directory.
For English version, the related memories for each testing dialogue are denoted as "relevant-id" in MADial-Bench/data/en/MADial-Bench-en-dialogue.json, with testing turns are written in "test-turn". 
For Chinese version, relavant memories are marked as "relevant-id" in MADial-Bench/data/zh/MADial-Bench-zh-dialogue.json. 
"relevant-id" are index of related memories in memory archieve, which are MADial-Bench/data/en/MADial-Bench-en-memory.json and MADial-Bench/data/en/MADial-Bench-en-memory.json.

The embeddings of dialogues and memories in memory recall task are in **embeddings** directory.

The inference results from LLM models in memory recognition and response genereation task are in **output** directory.

Evaluation results are in **results** directory with annotation files in **annotatation** directory. The critiera and guidelines are 

You can already make use of the benchmark if you read the paper and have strong coding ability. 

## Usage & start up

For memory recall task:

1. First download embeddings models and save them in **pretrained_models**.
2. Then run ```Embeddings.py``` to generate embeddings for dialogues and memories. 
3. Run ```embeddings_top_20_new.py``` to get top 20 candidates.
4. Run ```embedding_scores_new.py``` to calculate scores of certain metrics.

For memory recognition and response generation task:

0. for English version, run ```make_setting_candidates.py``` to generate dialogues for setting2 and 3.
1. First download opensourced LLM and save them in **pretrained_models**. If you usage API, then skip.
2. change the code in ```infer_setting1/2/3_en/ch.py``` to load your LLM, then run the infer program.
3. copy the output file path and change the path in ```evaluate.py``` to run automatic evaluation. It is not reliable, we recommand you to run step 4.
4. Human evaluation. Criteria and guidelines are in **annotation** directory. If you want to try LLM as judge, please try. We find the LLMs (up to 2024.10.25) are unable to do such a careful job.

Generate a file from the inference results of different LLMs:

1. run ```make_annotation_candidates.py``` to group the dialogus and the responses togather.
2. run ```prepare_anno.py``` to sample certain amount of dialogues from the test set and form an excel.



The codes are in a mess temporary, which needs to rewrite if you need one-click start codes. I am undergoing a family accident and will go back to work soon. Sorry for the inconvenience and I will tidy them up ASAP.

Please feel free to ask any questions and report issues.


## Please Cite
```
@inproceedings{he-etal-2025-madial,
    title = "{MAD}ial-Bench: Towards Real-world Evaluation of Memory-Augmented Dialogue Generation",
    author = "He, Junqing  and
      Zhu, Liang  and
      Wang, Rui  and
      Wang, Xi  and
      Haffari, Gholamreza  and
      Zhang, Jiaxing",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.499/",
    pages = "9902--9921",
    ISBN = "979-8-89176-189-6"
}


@misc{he2024madialbenchrealworldevaluationmemoryaugmented,
      title={MADial-Bench: Towards Real-world Evaluation of Memory-Augmented Dialogue Generation}, 
      author={Junqing He and Liang Zhu and Rui Wang and Xi Wang and Reza Haffari and Jiaxing Zhang},
      year={2024},
      eprint={2409.15240},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.15240}, 
}
```






