package fss;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;

/**
 * FSS (Feature Set Selection) Algorithm is the combination of the 
 * evaluator and searcher used for feature set selection. 
 */
public class FSS_Strategy {

   private ASEvaluation evaluation;
   private ASSearch search;
   private String searchDescription;
   private String evaluationDescription;
   private String attrUsed;
   public FSS_Strategy(ASEvaluation evaluation, ASSearch search) {
      super();
      this.evaluation = evaluation;
      this.search = search;
   }



   public FSS_Strategy(ASEvaluation evaluation, ASSearch search,
         String searchDescription, String evaluationDescription,
         String attrUsed) {
      super();
      this.evaluation = evaluation;
      this.search = search;
      this.searchDescription = searchDescription;
      this.evaluationDescription = evaluationDescription;
      this.attrUsed = attrUsed;
   }



   public ASEvaluation getEvaluation() {
      return evaluation;
   }

   public ASSearch getSearch() {
      return search;
   }

   public String getSearchName() {
      if (this.search != null) {
      return this.search.getClass().toString();
      } else {
         return "none";
      }
   }
   
   public String getEvaluationName() {
      if (this.evaluation != null) {
      return this.evaluation.getClass().toString();
      } else {
         return "none";
      }
   }
   
   public String getSearchDescription() {
      if (searchDescription == null) {
         return "default";
      } else {
         return searchDescription;
      }
   }
   
   public String getEvalutionString() {
      return this.getClass().toString();
   }

   public String getEvaluationDescription() {
      if (evaluationDescription == null) {
         return "default";
      } else {
         return evaluationDescription;
      }
   }



   public String getAttrUsed() {
      return attrUsed;
   }



   public void setAttrUsed(String attrUsed) {
      this.attrUsed = attrUsed;
   }





}
