### This is the online repository of the GAN-based test case augmentation Techniques.
## Task Definition

GAN-based Model-domain Failing Test Augmentation for Fault Localization.

## Dataset

The datasets we used are from [Defects4J](Defects4J,http://defects4j.org),[ManyBugs](http://repairbenchmarks.cs.umass.edu/ManyBugs/),[SIR](http://sir.unl.edu/portal/index.php).

This is a demo for nanoxml_v2, fault 1
source code:StdXMLParser.java
buggy line:363 if (! XMLUtil.checkLiteral(this.reader, "CDATA")) 
correct format:if (! XMLUtil.checkLiteral(this.reader, "CDATA["))

## Data Format

1. Coverage_Info/componentinfo.txt is stored in txt format. The first row is the number of total executed statements. The second row is the line number of each statement.

2. Coverage_Info/covMatrix.txt is stored in txt format. It is the model-domain test cases. Each row is a test case, the element 1 or 0 is the coverage information of statements, in which 1 denotes the corresponding statement is executed by the test case, and 0 otherwise.

3. Coverage_Info/error.txt is stored in txt format. It is the test cases' results, each line represents one result. 1 means a failed test case and 0 denotes a successful test case.

4. Coverage_Info/covMatrix_new.txt is stored in txt format. It is result of the generated model-domain test cases, each row represents one model-domain test case. the elements are the value of statements.

## Usage
You can get generated model-domain test cases' results using the following command.

for the code of example
```
cd example
python testcase_aug.py dev
```
for the code of nanoxml_v2, fault 1

```
cd nanoxml_v2_f1
python testcase_aug.py dev
```
Result is Coverage_Info/covMatrix_new.txt

## Dependency

- python version: python3.7.6
- pip install torch
- pip install numpy
## Evaluator
You can get the evaluated results using the following command.
```
cd nanoxml_v2_f1
cd SFL_compute_original
./run_original.sh
```
The file result.txt is the Rank and Exam of the bug statement.
