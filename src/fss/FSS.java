package fss;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.ObjectInputStream.GetField;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.ClassifierSubsetEval;
import weka.attributeSelection.ConsistencySubsetEval;
import weka.attributeSelection.CostSensitiveAttributeEval;
import weka.attributeSelection.CostSensitiveSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.HoldOutSubsetEvaluator;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.RaceSearch;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.SVMAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.attributeSelection.UnsupervisedAttributeEvaluator;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.AODE;
import weka.classifiers.bayes.WAODE;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.Prism;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.m5.CorrelationSplitInfo;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class FSS {

   /**
    * loads the given ARFF file and sets the class attribute as the last
    * attribute.
    *
    * @param filename    the file to load
    * @throws Exception  if somethings goes wrong
    */
   protected static Instances load(String filename) throws Exception {
      Instances       result;
      BufferedReader  reader;

      reader = new BufferedReader(new FileReader(filename));
      result = new Instances(reader);
      result.setClassIndex(result.numAttributes() - 1);
      reader.close();
      RaceSearch search = new RaceSearch(); 
      return result;
   }

   /**
    * saves the data to the specified file
    *
    * @param data        the data to save to a file
    * @param filename    the file to save the data to
    * @throws Exception  if something goes wrong
    */
   protected static void save(Instances data, String filename) throws Exception {
      BufferedWriter  writer;

      writer = new BufferedWriter(new FileWriter(filename));
      writer.write(data.toString());
      writer.newLine();
      writer.flush();
      writer.close();
   }



   private static List<Classifier> getClassifierList() {
      Classifier[] classifierList = {
            new RandomForest(),
            new Prism(),
            new SMO(),
            new AODE(),
            new DecisionTable()

      };
      return Arrays.asList(classifierList);
   }

   /**
    * Generate the list of FSS strategies (combinations of evaluator and search)
    * @return the list of strategies to use
    * @throws Exception 
    */
   private static List<FSS_Strategy> getFSSStrategyList(Instances inputTrain) throws Exception {

      List<FSS_Strategy> strategyList = new ArrayList<FSS_Strategy>();

      //add regular old classifier with no selection or search
//      strategyList.add(new FSS_Strategy(null, null));

      //ADDING RANKERS
      ASEvaluation[] attributeList = {
            new ChiSquaredAttributeEval(),
            new GainRatioAttributeEval(),
            new InfoGainAttributeEval(),
            new SymmetricalUncertAttributeEval(),
      };
      for (ASEvaluation eval : attributeList) {
         for (int i = 5; i < 35; i+=5) {

            Ranker ranker = new Ranker();
            ranker.setGenerateRanking(true);
            ranker.setThreshold(9999999.0);
            ranker.setNumToSelect(i);
            eval.buildEvaluator(inputTrain);
            String attrUsed = arrToString(ranker.search(eval, inputTrain), i);
            FSS_Strategy newStrategy = new FSS_Strategy(eval, 
                  ranker,
                  null,
                  "select_" + i,
                  attrUsed
                  );
            strategyList.add(newStrategy);
         }
      }
      
      //ADD SUBSET EVALS
      ASEvaluation[] subsetList = {
            new CfsSubsetEval(),
            new ConsistencySubsetEval(),
            };
      for (ASEvaluation subsetEval : subsetList) {
         GreedyStepwise greedySearch = new GreedyStepwise();
         subsetEval.buildEvaluator(inputTrain);
         int[] usedIndices = greedySearch.search(subsetEval, inputTrain);
         String attrUsed = arrToString(usedIndices, usedIndices.length);
         FSS_Strategy newStrategy = new FSS_Strategy(
               subsetEval,
               greedySearch,
               null,
               null,
               attrUsed);
         strategyList.add(newStrategy);
      }
      
      return strategyList;
   }


   public static void printResults(Evaluation evaluation, FSS_Strategy strategy, Classifier classifier) {
      System.out.printf("%s,%s,%s,%s,%s,%s,%f,%f,%f,%f\n",
            strategy.getEvaluationName(),
            strategy.getEvaluationDescription(),
            strategy.getSearchName(),
            strategy.getSearchDescription(),
            strategy.getAttrUsed(),
            classifier.getClass().toString(),
            evaluation.confusionMatrix()[0][0],
            evaluation.confusionMatrix()[0][1],
            evaluation.confusionMatrix()[1][0],
            evaluation.confusionMatrix()[1][1]
            );
   }

   public static String allIndicesString(int numAttr) {
      String s = "";
      for (int i = 0; i < numAttr; i++) {
         s += i;
         if (i != numAttr - 1) {
            s += "-";
         }
      }
      return s;
   }

   public static String arrToString(int[] arr, int numAttrUsed) {
      String s = "";
      for (int i = 0; i < numAttrUsed; i++) {
         s += arr[i];
         if (i != numAttrUsed - 1) {
            s += "-";
         }
      }
      return s;
   }


   public static void main(String[] args) throws Exception {     
      Instances     inputTrain;
      Instances     inputTest;
      // load data (class attribute is assumed to be last attribute)
      inputTrain = load(args[0]);
      inputTest  = load(args[1]);

      //uncomment for discretization
      //      Discretize filter = new Discretize();
      //      filter.setInputFormat(inputTrain);

      AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
      List<FSS_Strategy> strategyList = getFSSStrategyList(inputTrain);
      List<Classifier> classifierList = getClassifierList();
      System.out.println("evaluator,evaluatorOptions,search,searchOptions,"
            + "indices_used,classifier_name,true_positives,false_negatives,false_positives,true_negatives");

      for (Classifier baseClassifier: classifierList) {
         Evaluation evaluation = new Evaluation(inputTrain);
         baseClassifier.buildClassifier(inputTrain);
         evaluation.evaluateModel(baseClassifier, inputTest);
         printResults(evaluation, new FSS_Strategy(null, null), baseClassifier);
      }
      
      for (FSS_Strategy strategy : strategyList) {
         if (strategy.getSearch() != null && strategy.getEvaluation() != null) {
            classifier.setSearch(strategy.getSearch());
            classifier.setEvaluator(strategy.getEvaluation());
         }
         for (Classifier baseClassifier : classifierList) {
            Evaluation evaluation = new Evaluation(inputTrain);
            //evaluate
               classifier.setClassifier(baseClassifier);
               classifier.buildClassifier(inputTrain);
               evaluation.evaluateModel(classifier, inputTest);
               printResults(evaluation, strategy, baseClassifier);



            //print results
            //            System.out.println(evaluation.toSummaryString());
            //            System.out.println(evaluation.toMatrixString());

         }
      }
   }
}
