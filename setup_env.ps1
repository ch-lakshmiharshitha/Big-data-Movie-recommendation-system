$env:SPARK_HOME="D:\sem7\BigData\spark\spark-4.2.0-bin-hadoop3"

$env:PYTHONPATH="$env:SPARK_HOME\python;$env:SPARK_HOME\python\lib\pyspark.zip;$env:SPARK_HOME\python\lib\py4j-0.10.9.9-src.zip"

Write-Host "Spark environment configured."
Write-Host "SPARK_HOME = $env:SPARK_HOME"

python -c "import pyspark; print('PySpark:', pyspark.__version__)"