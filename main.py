import os

from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder\
    .master('local[1]')\
    .config("spark.driver.memory", "8g")\
    .appName('tcc_iaa')\
    .getOrCreate()

MODEL_DIR = 'D:\\TCC_IAA'
DATA_DIR = 'C:\\Users\\mathe\\Desktop\\TCC_IAA\\teste'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'validation')


def get_dataframe(data_path):
    df = None
    for class_num, class_name in enumerate(os.listdir(data_path)):
        file_names = os.listdir(os.path.join(data_path, class_name))
        for file_name in tqdm(file_names):
            temp_df = spark.read.csv(os.path.join(data_path, class_name, file_name), inferSchema=True)
            assembler = VectorAssembler(inputCols=temp_df.columns, outputCol="features")
            temp_df = assembler.transform(temp_df).select('features')
            temp_df = temp_df.withColumn('label', lit(class_num))

            if df is None:
                df = temp_df
            else:
                df = df.union(temp_df)

    return df

df_train = get_dataframe(TRAIN_DIR)
df_test = get_dataframe(TEST_DIR)
print((df_train.count(), len(df_train.columns)))
print((df_test.count(), len(df_test.columns)))

#lsvc = LinearSVC(maxIter=10, regParam=0.1)
svm = LinearSVC()
ovr = OneVsRest(classifier=svm)
lsvcModel = ovr.fit(df_train)
lsvcModel.save(os.path.join(MODEL_DIR, 'LsvcModel'))

transformed = lsvcModel.transform(df_test)

# Instantiate metrics object
metrics = MulticlassMetrics(transformed)

# Overall statistics
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)
