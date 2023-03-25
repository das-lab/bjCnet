# bjCnet

This is the research code for the paper "bjCnet: A Contrastive Learning-Based Framework for Method-Level Software Defect Prediction"

## Structure

- BugPrediction_Java: used to parse Java source code, generate AST and MSG.

- BugPrediction_Python: used to train bjCnet model.

## Usage

1. Unzip the `source code` dir in PROMISE Dataset. It need to note that the version number only keep one ".", and the project name need to be uniform, such as: jakarta-ant and apache-ant.

2. Instantiate the Process4PROMISE class in `BugPrediction_Python\scripts\DataPreprocess.py`, call `process` method to generate raw dataset.

3. Run `BugPrediction_Java\src\main\java\scripts\RunJoern.java` to generate AST.

4. Run `BugPrediction_Python\scripts\remove_built_in_dot.py` to remove dot files belong to built-in method.

5. Run `BugPrediction_Java\src\main\java\util\MSGBuilder.java` to generate MSG.

6. Instantiate the Process4MSG class in `BugPrediction_Python\scripts\DataPreprocess.py`, call `process` method to generate generate training dataset.

7. Run `BugPrediction_Python\run.py` to build bjCnet model.
