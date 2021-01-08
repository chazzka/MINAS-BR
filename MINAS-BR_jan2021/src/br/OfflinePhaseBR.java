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
    
    
    /**
     * representa o classificador. Esse Objeto é responsável pela fase offline do algoritmo, ele é responsável pelo treinamento.
     * @param trainingFile  caminho do arquivo do dataset de treino
     * @param classesConhecidas
     * @param fileOff
     * @param outputDirectory
     * @throws Exception 
     */
    public OfflinePhaseBR(ArrayList<Instance> trainingFile,
            double k_ini, 
            Set<String> classesConhecidas, 
            FileWriter fileOff, String outputDirectory) throws Exception{
        super("kmeans", fileOff, outputDirectory);
        this.setK_ini(k_ini);
        this.setTrainingData(trainingFile);
        this.training();
        fileOff.write("Label Cardinality: " + this.model.getCurrentCardinality() + "\n");
        fileOff.close();
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
        
        model = new Model();
        model.inicialize(super.getDirectory());
        Set<String> knownClassesLabels = new HashSet<>();
        //generate a set of micro-clusters for each class from the training set
        for(Map.Entry<String, ArrayList<Instance>> entry : super.getTrainingData().entrySet()) {
            String key = entry.getKey();
            ArrayList<Instance> subconjunto = entry.getValue();
            ArrayList<MicroCluster> clusterSet = null;
            int[] clusteringResult = new int[subconjunto.size()];
            // if the clustering algorithm is clustream
            if (getAlgOff().equals("clustream")) {
                System.out.println("Clustream");
                clusterSet = super.criarmodeloCluStreamOffline(subconjunto, key, (int) Math.ceil(subconjunto.size() * k_ini));
            // if the clustering algorithm is kmeans
            } else if (getAlgOff().equalsIgnoreCase("kmeans")) {
                System.out.println("Kmeans");
               // String aux = "datasets/mediamill/mediamill-train.arff[31, 33, 67].arff";
                clusterSet = super.createModelKMeansOffline(subconjunto, key, clusteringResult, (int) Math.ceil(subconjunto.size() * k_ini));
//                clusterSet = OnlinePhaseUtils.createModelKMeansLeader(subconjunto, clusteringResult, 1.25, 0);
            }
            model.getModel().put(key, clusterSet);
            knownClassesLabels.add(key);
//            model.setKnownClasses(key);
            System.out.println("Class: " + key + " size: " + clusterSet.size() + " n:" + subconjunto.size());
            super.getFileOut().write("Class: " + key + " size: " + clusterSet.size() + " n:" + subconjunto.size() + "\n");
        }
        model.setClasses(knownClassesLabels);
//        FreeChartGraph graphic = new FreeChartGraph(super.getDirectory(), "Offline Graph", super.getTrainingData().size());
//        graphic.createMicroClustersPlotOffPhase(model.getModel(), super.getDirectory());
    }
    
    /**
     * Separa os exemplos em subconjuntos, um para cada classe
     * @param D conjunto de treino
     * @param classesConhecidas
     * @throws Exception 
     */
    public void setTrainingData(ArrayList<Instance> D) throws Exception{
        int qtdeRotulos = 0;

        HashMap<String, ArrayList<Instance>> trainingData = new HashMap<String, ArrayList<Instance>>();

        for (int i = 0; i < D.size(); i++) {
            Set<String> labels = DataSetUtils.getLabelSet(D.get(i)); 
            qtdeRotulos += labels.size();
            ArrayList<Instance> generic; 
            
            //For each label assigned to an example, add this example into it repesctive set.
            for (String label : labels) { 
                try{
                    model.getMtxLabelsFrequencies().put(label, model.getMtxLabelsFrequencies().get(label) + 1);
                }catch(NullPointerException e){
                    model.getMtxProbabilities().put(label, 0.0);
                    model.getMtxLabelsFrequencies().put(label, 1);
                }
                
                ArrayList<Instance> dataset = null;
                try{
                    dataset = trainingData.get(label); 
                }catch(NullPointerException e){
                    System.out.println("Create new set for label: " + label);
                    dataset = new ArrayList<>();
                }
                dataset.add(D.get(i));
                trainingData.put(label,dataset);
            }
        }
        
        super.setTrainingData(trainingData);
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
