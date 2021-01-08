/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br;

import NoveltyDetection.KMeansMOAModified;
import NoveltyDetection.MicroCluster;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import dataSource.DataSetUtils;
import evaluate.Evaluator;
import evaluate.EvaluatorBR;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import utils.OnlinePhaseUtils;
import utils.Voting;

public class OnlinePhaseBR extends OnlinePhase {

    private int cardinalidadeAtual;
    private int lastCheck = 0;			// last time the clustering algorithm was executed
    private final double threshold;
    private int qtdeExExcluidos = 0;
    private int exShortTimeMem = 0;
    private ArrayList<Integer> timeStampExtension = new ArrayList<>();
    private ArrayList<Integer> timeStampNP = new ArrayList<>();
    private FileWriter extInfo;

    public OnlinePhaseBR(int cardinality, int numExNoveltyDetection, double threshold, String outputDirectory, FileWriter fileOn, String algOn) throws Exception {
        super(algOn, numExNoveltyDetection, outputDirectory, fileOn);
        this.cardinalidadeAtual = cardinality;
        this.threshold = threshold;
        extInfo = new FileWriter(new File(outputDirectory+"/extInfo.txt"),false);
    }
    
    /**
     * Classifies or rejects new examples
     * @param model
     * @param av
     * @param data
     * @param noveltyPatterns
     * @param fileOut
     * @throws IOException 
     */
    public void classify(Model model, EvaluatorBR av, Instance data, ArrayList<String> noveltyPatterns, FileWriter fileOut) throws IOException {
        Set<String> labels = DataSetUtils.getLabelSet(data); //get true labels
        model.verifyConceptEvolution(labels, super.getTimestamp());
        ArrayList<Voting> voting = identifyExample(data, model); //get predict labels
        
        //information about the classification
        String information = "Ex: " + super.getTimestamp() + "\t True Labels: " + labels.toString() + "\t MINAS-BR Prediction: ";
        
        //An example is consider unknown when it is outside all of micro-clusters' models
//        if (voting.size() > this.cardinalidadeAtual) {
        if (!voting.isEmpty()) {
            //classifies
            //Set<String> Z = av.thresholding(voting, this.getCardinalidadeAtual());
            Set<String> Z = av.thresholding(voting, voting.size());
            model.addPrediction(labels, Z);

            //write in the file
            String textOut = information;
            fileOut.write(textOut + Z.toString());
            fileOut.write("\n");
        } else {
            //rejects
            model.addPrediction(labels, null);
            
            //********** Example not explained by the current decision model: marked as unknown
            information = information + "unknwon";
            fileOut.write(information);
            fileOut.write("\n");
            exShortTimeMem++;

            //adding the example to the unkown memory 
            model.getShortTimeMemory().add(data, super.getTimestamp());

            //********** Searching for new valid micro-clusters created from unknown examples        	
            if ((model.getShortTimeMemory().size() >= this.getNumExNoveltyDetection()) && (this.getLastCheck() + this.getNumExNoveltyDetection() < this.getTimestamp())) {
                System.out.println("************Novelty Detection Phase***********");
                noveltyPatternProcedure(model,noveltyPatterns,fileOut,av);
            }
        }
    }

    /**
     * Detects if the new micro-clusters are extensions or NPs.
     *
     * @param model
     * @param av
     * @param noveltyPatterns lista com os NPs
     * @param fileOut arquivo para a saida dos resultados
     * @throws IOException
     */
    public void noveltyPatternProcedure(Model model, ArrayList<String> noveltyPatterns, FileWriter fileOut, EvaluatorBR ev) throws IOException {
//        int numMinExCluster = 10;
        int numMinExCluster = 3;
        this.setLastCheck(super.getTimestamp());
        String textoArq = "";
        
        //vector indicating the examples to be removed from short-time-memory because they were used to create a new valid micro-cluster                
        int[] removeExamples = new int[model.getShortTimeMemory().size()];
        int k = (int) (model.getShortTimeMemory().size() * 0.1); //obsolete when apply k-means+leader 
        
        //temporary new micro-cluster formed by unknown examples
        ArrayList<MicroCluster> model_unk = super.createModelFromExamples(model.getShortTimeMemory().getData(), k, removeExamples, 
                OnlinePhaseUtils.getMaxRadius(model.getModel()));

        ArrayList<MicroCluster> newMicroClusters = new ArrayList<MicroCluster>();
        
        //for each candidate micro-cluster
        for (int temp_count_k = 0; temp_count_k < model_unk.size(); temp_count_k++) {
            MicroCluster model_unk_aux = model_unk.get(temp_count_k);
            ArrayList<Instance> toClassify = new ArrayList<>(); //stores for classification examples which will form the new micro-clusters 
            ArrayList<MicroCluster> extModels = new ArrayList<>();
            ArrayList<MicroCluster> novModels = new ArrayList<>();
            //Representative validation
            if ((!model_unk_aux.isEmpty()) && (model_unk_aux.getWeight() >= numMinExCluster)) {
                //For each BR-model
                for (Map.Entry<String, ArrayList<MicroCluster>> listaMicroClusters : model.getModel().entrySet()) {
                    //Silhouette validation
                    if (this.clusterValidationSilhouette(model_unk_aux, listaMicroClusters.getValue()) == true) {
                        //********** The new micro-cluster is valid ****************
                        for (int count_remove = 0; count_remove < removeExamples.length; count_remove++) {
                            // mark the examples to be removed with the label -2
                            if (removeExamples[count_remove] == temp_count_k) {
                                removeExamples[count_remove] = -2;
                                toClassify.add(model.getShortTimeMemory().getData().get(count_remove)); //add to classify
                            }
                        }
                        //identifies the closer micro-cluster to the new valid micro-cluster 
//                        ret_func[0] <- posMinDist
//                        ret_func[1] <- minDist
//                        ret_func[0] <- threshold                        
                        double[] ret_func = identifyCloserMicroClusterTV1(model_unk_aux, listaMicroClusters.getValue(), this.getThreashold());
//                        Voting ret_func = identifyCloserMicroClusterTV2(model_unk_aux, listaMicroClusters.getValue(), this.getThreashold());
//                        Voting ret_func = identifyCloserMicroClusterTV3(model_unk_aux, listaMicroClusters.getValue(), this.getThreashold());
                        //if dist < (stdDev * f) than extension, otherwise NP
                        if (ret_func[1] < ret_func[2]) {
                            textoArq = "Thinking " + "extension: " + "C " + listaMicroClusters.getKey() + " - " + (int) model_unk_aux.getWeight() + " examples";
                            extModels.add(listaMicroClusters.getValue().get((int)ret_func[0]));
                        } else {
                            textoArq = "Thinking " + "np: " + "N " + listaMicroClusters.getKey() + " - " + (int) (int) model_unk_aux.getWeight() + " examples";
                            novModels.add(listaMicroClusters.getValue().get((int)ret_func[0]));
                        }
                    } //End of micro-cluster validation
                } //End of all models
                
                Set<String> labelSet = new HashSet<>();
                if (!novModels.isEmpty() || !extModels.isEmpty()) {
                    //If the number of models which considered the new micro-cluster as NP is greater than label cardinality
//                    if (extModels.size() > 0) {
                    if (extModels.size() > this.getCardinalidadeAtual()) {
                        //Extension
                        this.getExtInfo().write("*********Extension**********"+"\n");
                        this.getExtInfo().write("Timestamp"+ this.getTimestamp() + "\n");
                        this.timeStampExtension.add(this.getTimestamp());
                        this.getTimeStampExtension().add(this.getTimestamp());
                        int i = 0;
                        while ( i < extModels.size()) {
                            if ((extModels.get(i).getCategory().equalsIgnoreCase("normal")) || (extModels.get(i).getCategory().equalsIgnoreCase("ext"))) {
                                //extension of a class learned offline
                                newMicroClusters.add(this.updateModel(model_unk_aux, model.getModel().get(extModels.get(i).getLabelClass()), extModels.get(i).getLabelClass(), "ext"));
                                textoArq = textoArq.concat("Thinking " + "Extension: " + "C " + extModels.get(i).getLabelClass() + " - " + (int) model_unk_aux.getWeight() + " examples" + "\n");
                            } else {
                                //extension of a novelty pattern
                                newMicroClusters.add(this.updateModel(model_unk_aux, model.getModel().get(extModels.get(i).getLabelClass()), extModels.get(i).getLabelClass(), "extNov"));
                                textoArq = textoArq.concat("Thinking " + "NoveltyExtension: " + "N " + extModels.get(i).getLabelClass() + " - " + (int) model_unk.get(temp_count_k).getWeight() + " examples");
                            }
                            labelSet.add(extModels.get(i).getLabelClass());
                            i++;
                            this.getExtInfo().write(textoArq+"\n");
                        }
                    } else {
                        //novelty pattern
                        this.getExtInfo().write("*********Novelty Pattern**********"+"\n");
                        this.getExtInfo().write("Timestamp"+ this.getTimestamp() + "\n");
                        this.timeStampNP.add(this.getTimestamp());
                        
                        for (MicroCluster mc : extModels) {
                            labelSet.add(mc.getLabelClass());
                        }
                        labelSet.add("NP" + Integer.toString(noveltyPatterns.size() + 1));
                        extModels.add(model_unk_aux);
                        this.createModel(model, extModels, noveltyPatterns, textoArq);
//                        newMicroClusters.add(model_unk_aux);
//                        NPlist.put(model_unk_aux, extModels);
                    }
                    for (Instance inst : toClassify) {
                        model.addPrediction(DataSetUtils.getLabelSet(inst), labelSet);
//                        try{
                            model.removerUnknown(labelSet);
//                        }catch(Exception e){
//                            System.out.println("");
//                        }
                    }
                } //end of mininum number of examples     
            } //end of each valid cluster
        }
//        NDgraph.plotNoveltyDetection(this.getTimestamp(), model.getModel(), newMicroClusters, true);
        System.out.println(textoArq);
        fileOut.write(textoArq);
        fileOut.write("\n");
        //remove the examples marked with label -2, i. e., the unknown examples used in the creation of new valid micro-clusters
        int count_rem = 0;
        for (int g = 0; g < removeExamples.length; g++) {
            if (removeExamples[g] == -2) {
                model.getShortTimeMemory().getData().remove(g - count_rem);
                model.getShortTimeMemory().getTimestamp().remove(g - count_rem);
                count_rem++;
//                qtdeExExcluidos++;
            }
        }
    }

    /**
     * Updates the models adding the new micro-clusters
     *
     * @param modelUnk micro-grupo novidade
     * @param model modelo a ser atualizado
     * @param classLabel rotulo do modelo
     * @param category tipo de micro-grupo (ext, nov,...)
     */
    public MicroCluster updateModel(MicroCluster modelUnk, ArrayList<MicroCluster> model, String classLabel, String category) {
        //update the decision model by adding new valid micro-clusters
        MicroCluster microCluster = new MicroCluster(modelUnk, classLabel, category, this.getTimestamp());
//        modelUnk.setCategory(category);
//        modelUnk.setLabelClass(classLabel);
        model.add(microCluster);
        return microCluster;
    }

    /**
     * creates a new model to represent the new Novelty Pattern
     *
     * @param model
     * @param novidade - novelty
     * @param extModels - models witch consider the novelty like extesion
     * @param label - new label to assign
     * @param category - category to assign
     */
    public void createModel(Model model, MicroCluster novidade, ArrayList<Voting> extModels, String label, String category) {
        ArrayList<MicroCluster> newModel = new ArrayList<>();
//        Collections.sort(novModels);
        int i = 0;

        while (i < extModels.size()) {
            //Adiciona os z (cardinalidadeAtual) micro-grupos dos modelos mais próximos
            ArrayList<Integer> ret_func = getClosestsNPMicroClusters(novidade, model.getModel().get(extModels.get(i).getKey()));
            MicroCluster mic = new MicroCluster(model.getModel().get(extModels.get(i).getKey()).get(ret_func.get(i)), label, category, super.getTimestamp());
            newModel.add(mic);
            i++;
        }
        novidade.setCategory(category);
        novidade.setLabelClass(label);
        newModel.add(novidade);
        model.getModel().put(label, newModel);
    }

    /**
     * creates a new model to represent the new Novelty Pattern
     * @param model
     * @param NPlist
     * @param noveltyPatternsControl
     * @param textoArq
     * @throws java.io.IOException
     */
    public void createModel(Model model, HashMap<MicroCluster, ArrayList<Voting>> NPlist, ArrayList<String> noveltyPatternsControl, String textoArq) throws NumberFormatException, IOException {
        double maxD = getMaxDiameter(NPlist.keySet());
        HashMap<String, ArrayList<MicroCluster>> NPlistClustering = OnlinePhaseUtils.KMeansLeader(NPlist.keySet(), maxD);

        for (Map.Entry<String, ArrayList<MicroCluster>> entry : NPlistClustering.entrySet()) {
            String key = entry.getKey();
            ArrayList<MicroCluster> value = entry.getValue();
            ArrayList<MicroCluster> newModel = new ArrayList<MicroCluster>();
            for (MicroCluster novidade : value) {
                ArrayList<Voting> arrayList = NPlist.get(value);
                int i = 0;
                try{
                    for (Voting extModels : arrayList) {
                        ArrayList<Integer> ret_func = getClosestsNPMicroClusters(novidade, model.getModel().get(extModels.getKey()));
                        MicroCluster mic = new MicroCluster(model.getModel().get(extModels.getKey()).get(ret_func.get(i)), "NP" + Integer.toString(noveltyPatternsControl.size() + 1), "nov", super.getTimestamp());
                        newModel.add(mic);
                        i++;
                    }
                }catch(Exception e){
                    System.out.println("The NP doesn't have extensions");
                }
                novidade.setTime(super.getTimestamp());
                novidade.setCategory("nov");
                novidade.setLabelClass("NP" + Integer.toString(noveltyPatternsControl.size() + 1));
                newModel.add(novidade);
            }
            model.getModel().put("NP" + Integer.toString(noveltyPatternsControl.size() + 1), newModel);
            model.addNPs(super.getTimestamp());
            noveltyPatternsControl.add("NP" + Integer.toString(noveltyPatternsControl.size() + 1));
            textoArq = textoArq.concat("ThinkingNov: " + "Novidade " + "NP" + Integer.toString(noveltyPatternsControl.size() + 1) + " - " + newModel.size() + " micro-clusters" + "\n");
        }
    }
    /**
     * creates a new model to represent the new Novelty Pattern
     * @param model
     * @param novelty
     * @param noveltyPatternsControl
     * @param textoArq
     * @throws java.io.IOException
     */
    public void createModel(Model model, ArrayList<MicroCluster> novelty, ArrayList<String> noveltyPatternsControl, String textoArq) throws NumberFormatException, IOException {
        for (MicroCluster mc : novelty) {
            mc.setTime(super.getTimestamp());
            mc.setCategory("nov");
            mc.setLabelClass("NP" + Integer.toString(noveltyPatternsControl.size() + 1));
        }
        model.getModel().put("NP" + Integer.toString(noveltyPatternsControl.size() + 1), novelty);
        model.addNPs(super.getTimestamp());
        noveltyPatternsControl.add("NP" + Integer.toString(noveltyPatternsControl.size() + 1));
        textoArq = textoArq.concat("ThinkingNov: " + "Novelty " + "NP" + Integer.toString(noveltyPatternsControl.size() + 1) + " - " + novelty.size() + " micro-clusters" + "\n");
        this.getExtInfo().write(textoArq+"\n");
        System.out.println("ThinkingNov: " + "Novelty " + "NP" + Integer.toString(noveltyPatternsControl.size() + 1) + " - " + novelty.size() + " micro-clusters");
    }

    /**
     * Get micro-clusters of the models witch consider the NP as extension
     *
     * @param NP
     * @param modelo
     * @return
     */
    public ArrayList<Integer> getClosestsNPMicroClusters(MicroCluster NP, ArrayList<MicroCluster> modelo) {
        double distance = 0;
        double t = NP.getRadius() + Math.pow(NP.getRadius(), 2);
        ArrayList<Integer> closestMic = new ArrayList<Integer>();
        // calculates the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modelo.size(); i++) {
            distance = KMeansMOAModified.distance(modelo.get(i).getCenter(), NP.getCenter());
            if (distance < t) {
                closestMic.add(i);
            }
        }
        return closestMic;
    }

    /**
     * Identifies if the new micro-clusters are NP or Extension using the TV2
     * strategy (max distance between the closest micro-clusters and the others)
     *
     * @param modelUnk novidade
     * @param modelo modelo
     * @param threshold limiar estipulado pelo usuário
     * @return
     */
//    public ArrayList<Integer> getClosestMicroClustersFromExtensions(MicroCluster NP, ArrayList<MicroCluster> modelo, double threshold) {
//        double minDistance = 0;
//        int posMinDistance = 0;
//        double distance = 0;
//        ArrayList<Integer> listClosestMic = new ArrayList<Integer>();
//        // calculate the distance between the center of the new cluster to the existing clusters 
//        for (int i = 1; i < modelo.size(); i++) {
//            distance = KMeansMOAModified.distance(modelo.get(i).getCenter(), NP.getCenter());
//            double vthreshold = distance;
//            vthreshold = vthreshold / 2;
//            String categoria = modelo.get(posMinDistance).getCategory();
//            String classe = modelo.get(posMinDistance).getLabelClass();
//        }
//
//        return new Voting(categoria, classe, minDistance, vthreshold);
//    }

    /**
     * Identifica o micro-grupo mais próximo da novidade e calcula o limiar para
     * definir um NP
     *
     * @param modelUnk novidade
     * @param modelo modelo
     * @param threshold limiar estipulado pelo usuário
     * @return
     */
    public double[] identifyCloserMicroClusterTV1(MicroCluster modelUnk, ArrayList<MicroCluster> modelo, double threshold) {
        double[] ret = new double[3];
        double minDistance = Double.MAX_VALUE;
        int posMinDistance = 0;
        // calculate the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modelo.size(); i++) {
            double distance = KMeansMOAModified.distance(modelo.get(i).getCenter(), modelUnk.getCenter());
            if (distance < minDistance) {
                minDistance = distance;
                posMinDistance = i;
            }
        }
//        double vthreshold = (modelo.get(posMinDistance).getRadius()) * threshold;
        double vthreshold = modelo.get(posMinDistance).getRadius()/2 * threshold;
        if (minDistance < vthreshold) {
            ret[0] = posMinDistance;
            ret[1] = minDistance;
            ret[2] = vthreshold;
        } else { //intercept
            vthreshold = modelo.get(posMinDistance).getRadius() / 2 + modelUnk.getRadius() / 2;
            ret[0] = posMinDistance;
            ret[1] = minDistance;
            ret[2] = vthreshold;
        }
        return ret;
    }

    /**
     * Identifies if the new micro-clusters are NP or Extension using the TV2
     * strategy (max distance between the closest micro-clusters and the others)
     *
     * @param modelUnk novidade
     * @param modelo modelo
     * @param threshold limiar estipulado pelo usuário
     * @return
     */
    public Voting identifyCloserMicroClusterTV2(MicroCluster modelUnk, ArrayList<MicroCluster> modelo, double threshold) {
        double minDistance = Double.MAX_VALUE;
        int posMinDistance = 0;
        // calculates the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modelo.size(); i++) {
            double distance = KMeansMOAModified.distance(modelo.get(i).getCenter(), modelUnk.getCenter());
            if (distance < minDistance) {
                minDistance = distance;
                posMinDistance = i;
            }
        }
        double vthreshold = minDistance;
        for (int i = 1; i < modelo.size(); i++) {
            double distance = KMeansMOAModified.distance(modelo.get(i).getCenter(), modelo.get(posMinDistance).getCenter());
            if (distance > vthreshold) {
                vthreshold = distance;
            }
        }
        String categoria = modelo.get(posMinDistance).getCategory();
        String classe = modelo.get(posMinDistance).getLabelClass();

        return new Voting(categoria, classe, minDistance, posMinDistance, vthreshold);
    }
    /**
     * Identifies if the new micro-clusters are NP or Extension using the TV3
     * strategy (mean between the Euclidian distance of the closest micro-clusters and the others)
     *
     * @param modelUnk novidade
     * @param modelo modelo
     * @param threshold limiar estipulado pelo usuário
     * @return
     */
    public Voting identifyCloserMicroClusterTV3(MicroCluster modelUnk, ArrayList<MicroCluster> modelo, double threshold) {
        double minDistance = Double.MAX_VALUE;
        int posMinDistance = 0;
        double distance = 0;
        double sumDistance = 0;
        for (int i = 1; i < modelo.size(); i++) {
            sumDistance += KMeansMOAModified.distance(modelo.get(i).getCenter(), modelUnk.getCenter());
        }
        double vthreshold = sumDistance / modelo.size();
        for (int i = 1; i < modelo.size(); i++) {
            distance = KMeansMOAModified.distance(modelo.get(i).getCenter(), modelo.get(posMinDistance).getCenter());
            if (distance < minDistance) {
                minDistance = distance;
                posMinDistance = i;
            }
        }
        String categoria = modelo.get(posMinDistance).getCategory();
        String classe = modelo.get(posMinDistance).getLabelClass();

        return new Voting(categoria, classe, minDistance, posMinDistance, vthreshold);
    }

    /**
     * Identifies which models explain an example
     *
     * @param data
     * @param model
     * @return 
     */
    public ArrayList<Voting> identifyExample(Instance data, Model model) {
        ArrayList<Voting> voting = new ArrayList<>();
        for (Map.Entry<String, ArrayList<MicroCluster>> listaMicroClusters : model.getModel().entrySet()) {
            double distance = 0.0;
            if (listaMicroClusters.getValue().size() > 0) {
                String key = listaMicroClusters.getKey();
                double minDist = Double.MAX_VALUE;
                int posMinDist = 0;
                int pos = 0;
                for (MicroCluster microCluster : listaMicroClusters.getValue()) {
                    double[] aux = Arrays.copyOfRange(data.toDoubleArray(), data.numOutputAttributes(), data.numAttributes());
                    Instance inst = new DenseInstance(1, aux);
                    distance = microCluster.getCenterDistance(inst);
                    if(distance < minDist){
                        minDist = distance;
                        posMinDist = pos;
                    }
                    pos++;
                }
                
                if (minDist <= listaMicroClusters.getValue().get(posMinDist).getRadius()) {
                    Voting result = new Voting();
                    result.setKey(key);
                    result.setCategory(listaMicroClusters.getValue().get(posMinDist).getCategory()); //Normal, extension ou novelty
                    result.setDistance(minDist);

                    /*add the examples in micro-clusters*/
                    //                        double[] aux = Arrays.copyOfRange(data.toDoubleArray(), this.qtdeTotalClasses, data.numAttributes());
                    //                        Instance inst = new DenseInstance(1, aux);
                    //                        listaMicroClusters.getValue().get(posMinDistance).insert(inst, this.timestamp);
                    //                    System.out.println("Classificado como: " + key + "\n");
                    
                    voting.add(result);

                    listaMicroClusters.getValue().get(posMinDist).setTime(this.getTimestamp());
                }
            }
        }
        return voting;
    }

    /**
     * @return the threshold
     */
    public double getThreashold() {
        return getThreshold();
    }

    /**
     * @return the qtdeExExcluidos
     */
    public int getQtdeExExcluidos() {
        return qtdeExExcluidos;
    }

    /**
     * Gets the biggest micro-clusters diameter
     *
     * @param keySet
     * @return
     */
    private double getMaxDiameter(Set<MicroCluster> keySet) {
        double maxD = 0;
        double maxD2 = 0;
        for (Iterator<MicroCluster> iterator = keySet.iterator(); iterator.hasNext();) {
            MicroCluster next = iterator.next();
            maxD = next.getRadius() * 2;
            if (maxD < (next.getRadius() * 2)) {
                
            }
        }
        return maxD;
    }
    
    public void removeOldMicroClusters(int windowSize, Model modelo, FileWriter fileOut) throws IOException {
        ArrayList<MicroCluster> listaMicro = new ArrayList<MicroCluster>();
        ArrayList<String> classToRemove = new ArrayList<>();
//        try{
            for (Map.Entry<String, ArrayList<MicroCluster>> entry : modelo.getModel().entrySet()) {
                String key = entry.getKey();
                ArrayList<MicroCluster> value = entry.getValue();
                for (int i = 0; i < value.size(); i++) {
                    if (value.get(i).getTime() < (super.getTimestamp() - (windowSize))) {
                        listaMicro.add(value.get(i));
                        super.getSleepMemory().add(value.get(i));
                        fileOut.write("Removed micro-clusters: " + i +" label: " + value.get(i).getLabelClass() + " category: " +  value.get(i).getCategory());
                        fileOut.write("\n");
                        value.remove(i);
                        i--;
                    }
                }
                try{
                    super.getFileOn().write("Timestamp: " + super.getTimestamp() + " - removed micro-clusters: " + listaMicro.size() + " - model's size ["+value.get(0).getLabelClass()+"]:" + value.size() + "\n");
                    System.out.println("Timestamp: " + super.getTimestamp() + " - removed micro-clusters: " + listaMicro.size() + " - model's size ["+value.get(0).getLabelClass()+"]:" + value.size());
                }catch(Exception e){
                    classToRemove.add(key);
                }
            }
            for (String classe : classToRemove) {
                modelo.getModel().remove(classe);
            }
    }
    

    /**
     * @return the exShortTimeMem
     */
    public int getExShortTimeMem() {
        return exShortTimeMem;
    }

    /**
     * @return the cardinalidadeAtual
     */
    public int getCardinalidadeAtual() {
        return cardinalidadeAtual;
    }

    /**
     * @param cardinalidadeAtual the cardinalidadeAtual to set
     */
    public void setCardinalidadeAtual(int cardinalidadeAtual) {
        this.cardinalidadeAtual = cardinalidadeAtual;
    }

    /**
     * @return the threshold
     */
    public double getThreshold() {
        return threshold;
    }

    /**
     * @return the lastCheck
     */
    public int getLastCheck() {
        return lastCheck;
    }

    /**
     * @param lastCheck the lastCheck to set
     */
    public void setLastCheck(int lastCheck) {
        this.lastCheck = lastCheck;
    }


    /**
     * @return the timeStampExtension
     */
    public ArrayList<Integer> getTimeStampExtension() {
        return timeStampExtension;
    }

    /**
     * @param timeStampExtension the timeStampExtension to set
     */
    public void setTimeStampExtension(ArrayList<Integer> timeStampExtension) {
        this.timeStampExtension = timeStampExtension;
    }


    /**
     * @return the timeStampNP
     */
    public ArrayList<Integer> getTimeStampNP() {
        return timeStampNP;
    }

    /**
     * @return the extInfo
     */
    public FileWriter getExtInfo() {
        return extInfo;
    }

}
