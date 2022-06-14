Sequência de execução dos scripts:

1. video_processor.py: realiza a leitura dos videos e divide em segmentos de videos menores, classificando-os conforme o arquivo de labels disponibilizado no site da Drive & act.
2. model_genereator.py: realiza o treinamento e gera um modelo de cnn para a base farm state. Esse passo não é necessário se já tiver modelo um modelo pré treinado.
3. feature_extractor: lê os segmentos de video gerados pelo video_processor, carrega o modelo de cnn gerado pelo model_generator, então extrai as caracterisitcas salvando em um csv.
4. classifier_sklearn.py: lê as caracteristicas extraídas e realiza treinamento e teste.

O arquivo classifier_spark.py foi criado para executar o passo 4 com a biblioteca sparkML em vez do sklearn, porém ele se mostrou menos eficiente para o nosso problema, então o arquivo serve apenas para consulta.