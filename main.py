import os

from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.master('local[1]').appName('tcc_iaa').getOrCreate()

MODEL_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA'
DATA_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\deep_features'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'validation')

df = None

for class_num, class_name in enumerate(os.listdir(TRAIN_DIR)):
    file_names = os.listdir(os.path.join(TRAIN_DIR, class_name))
    for file_name in tqdm(file_names):
        temp_df = spark.read.csv(os.path.join(TRAIN_DIR, class_name, file_name), inferSchema=True)
        assembler = VectorAssembler(inputCols=temp_df.columns, outputCol="features")
        temp_df = assembler.transform(temp_df).select('features')
        temp_df = temp_df.withColumn('label', lit(class_num))

        if df is None:
            df = temp_df
        else:
            df = df.union(temp_df)


print((temp_df.count(), len(temp_df.columns)))
lsvc = LinearSVC(maxIter=10, regParam=0.1)
lsvcModel = lsvc.fit(df)