/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br;

import NoveltyDetection.ClustreamKernelMOAModified;
import NoveltyDetection.KMeansMOAModified;
import NoveltyDetection.MicroCluster;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import dataSource.DataSetUtils;
//import evaluate.FreeChartGraph;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import moa.cluster.CFCluster;
import moa.cluster.Clustering;
import utils.OnlinePhaseUtils;

/**
 * 
 * @author joel
 * 
 */
public final class OfflinePhaseBR extends OfflinePhase{
    private Model model;
    private double k_ini;
    
    public OfflinePhaseBR(ArrayList<Instance> trainingFile,
            double k_ini, 
            FileWriter fileOff, 
            String outputDirectory) throws Exception{
        
        super("kmeans", fileOff, outputDirectory);
        this.setK_ini(k_ini);
        this.setTrainingData(trainingFile);
        this.training();
        fileOff.write("Label Cardinality: " + this.model.getCurrentCardinality() + "\n");
    }
    
    /**
     * Builds the model
     * @throws IOException
     * @throws Exception 
     */
    public void training() throws IOException, Exception {
        //create one training file for each problem class 
        System.out.print("Classes for the training phase (offline): ");
        super.getFileOut().write("Classes for the training phase (offline): ");
        System.out.println("" + super.getTrainingData().keySet().toString());
        super.getFileOut().write("" + super.getTrainingData().keySet().toString());
        System.out.print("\nQuantidade de classes: ");
        super.getFileOut().write("\nQuantidade de classes: ");
        System.out.println("" + super.getTrainingData().size());
        super.getFileOut().write("" + super.getTrainingData().size() + "\n");
        
        
        //generate a set of micro-clusters for each class from the training set
        for(Map.Entry<String, ArrayList<Instance>> entry : super.getTrainingData().entrySet()) {
            String key = entry.getKey();
            ArrayList<Instance> subconjunto = entry.getValue();
            ArrayList<MicroCluster> clusterSet = null;
            int[] clusteringResult = new int[subconjunto.size()];
            
            clusterSet = super.createModelKMeansOffline(subconjunto, 
                    key, 
                    clusteringResult,
                    (int) Math.ceil(subconjunto.size() * k_ini));
            
            model.getModel().put(key, clusterSet);
            System.out.println("Class: " + key + " size: " + clusterSet.size() + " n:" + subconjunto.size());
            super.getFileOut().write("Class: " + key + " size: " + clusterSet.size() + " n:" + subconjunto.size() + "\n");
        }
        model.setClasses(this.getTrainingData().keySet());
    }
    
    /**
     * Separa os exemplos em subconjuntos, um para cada classe
     * @param D conjunto de treino
     * @param classesConhecidas
     * @throws Exception 
     */
    public void setTrainingData(ArrayList<Instance> D) throws Exception{
        model = new Model();
        model.inicialize(super.getDirectory());
        int qtdeRotulos = 0;
        HashMap<String, ArrayList<Instance>> trainingData = new HashMap<String, ArrayList<Instance>>();

        for (int i = 0; i < D.size(); i++) {
            Set<String> labels = DataSetUtils.getLabelSet(D.get(i)); 
            qtdeRotulos += labels.size();
            ArrayList<Instance> generic; 
            
            //For each label assigned to an example, add this example into it repesctive set.
            for (String label : labels) { 
                ArrayList<Instance> dataset;
                try{
                    dataset = trainingData.get(label); 
                    dataset.add(D.get(i));
                }catch(NullPointerException e){
                    System.out.println("Create new set for label: " + label);
                    dataset = new ArrayList<>();
                    dataset.add(D.get(i));
                }
                trainingData.put(label,dataset);
                
                //Filling the matrix T (frequencies)
                for (String label_column : labels) { 
                    String mtxCordinate = label+","+label_column;
                    int frequency = 0;
                    try{
                        frequency = model.getMtxLabelsFrequencies().get(mtxCordinate);
                        frequency ++;
                        model.getMtxLabelsFrequencies().put(mtxCordinate, frequency);
                    }catch(NullPointerException e){
                        model.getMtxLabelsFrequencies().put(mtxCordinate, 1);
                    }
                }
            }
            this.model.incrementNumerOfObservedExamples();
        }
        super.setTrainingData(trainingData);
        this.model.setInitialProbabilities();
        this.model.setCurrentCardinality(Math.ceil(qtdeRotulos/D.size()));
    }
    

    /**
     * @return the model
     */
    public Model getModel() {
        return model;
    }

    /**
     * @param k_ini the k_ini to set
     */
    public void setK_ini(double k_ini) {
        this.k_ini = k_ini;
    }

}
