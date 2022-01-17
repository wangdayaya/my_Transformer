### introduce

It mainly implements the principle code of Transformer, and uses Transformer to achieve simple Chinese-English translation task.
### main environment

1. python3.7
2. tensorflow-gpu=1.14.0


### usage

1. dictionary or data processing

        python data_loader.py --config config/Hyperparameter.yml
    
2. train model

            python main.py --config config/Hyperparameter.yml --mode train
      
        print part of the training process
            
            INFO:tensorflow:train/enc_0 = multi - head   attention   允许 模型 共同 关注 来自 不同 位置 的 不同 演示 子 阵列 的 信息 。
            INFO:tensorflow:train/target_0 = MultI - head   attachments   allow   the   model   to   focus   jointly   on   information   from   different   presentation   subarras   at   different   locations . <\s>
             - train/pred_0 = MultI The     attachments   allow   put   the   model   focus   the   on   value   input   different   value   subarras   different   presentation   locations . <\s> on   information   different           different     on
            INFO:tensorflow:loss = 1.090371, step = 7600 (3.203 sec)
            INFO:tensorflow:global_step/sec: 30.9693
            ...
            INFO:tensorflow:train/enc_0 = 论文 的 标题 是 “ Attention   Is   All   You   Need ” 。
            INFO:tensorflow:train/target_0 = The   title   of   the   paper   is   ' Attention   Is   All   You   Need ' . <\s>
             - train/pred_0 = The   title   is   the   paper   '   ' Attention   Is   All   All   You ' . <\s>     .                                              
            INFO:tensorflow:loss = 0.43084797, step = 12200 (3.229 sec)
            INFO:tensorflow:global_step/sec: 31.1915
            ...
            INFO:tensorflow:train/enc_0 = 谷歌 研究 博客 显示 了 “ Coreference   Resolution ” 的 出色 表现 。
            INFO:tensorflow:train/target_0 = Google   Research   blogs   show   great   performance   with   the   ' Coreference   Resolution ' . <\s>
             - train/pred_0 = Google   Research   blogs   show   great   performance   with   the   ' Coreference   Resolution ' . <\s>   '                 Google       ' Google Google               ' '    
            INFO:tensorflow:loss = 0.07248693, step = 19900 (3.227 sec)
            INFO:tensorflow:Saving checkpoints for 20000 into logs/model.ckpt.
    
        
3. predict

    python predict.py --config config/Hyperparameter.yml --src "你好。我很高兴见到你。"

### Thank

    https://github.com/DongjunLee/transformer-tensorflow