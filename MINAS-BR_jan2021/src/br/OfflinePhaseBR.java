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
    private double cardinality;                                  //cardinalidade de rótulo do dataset
    private Set<String> classesConhecidas;
    private double k_ini;
    
    
    /**
     * representa o classificador. Esse Objeto é responsável pela fase offline do algoritmo, ele é responsável pelo treinamento.
     * @param trainingFile  caminho do arquivo do dataset de treino
     * @param classesConhecidas
     * @param fileOff
     * @param outputDirectory
     * @throws Exception 
     */
    public OfflinePhaseBR(ArrayList<Instance> trainingFile, double k_ini, Set<String> classesConhecidas, FileWriter fileOff, String outputDirectory) throws Exception{
        super("kmeans", fileOff, outputDirectory);
        this.setK_ini(k_ini);
        this.setTrainingData(trainingFile, classesConhecidas);
        this.setClassesConhecidas(classesConhecidas);
        this.training();
        fileOff.write("Label Cardinality: " + this.getCardinality() + "\n");
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
    public void setTrainingData(ArrayList<Instance> D, Set<String> classesConhecidas) throws Exception{
        int qtdeRotulos = 0;

        //K -> Classe
        //V -> Lista de exemplos de cada classe
        HashMap<String, ArrayList<Instance>> trainingData = new HashMap<String, ArrayList<Instance>>();
        Iterator iterator = classesConhecidas.iterator();
        while(iterator.hasNext()){ //Para cada classe uma lista é criada. Essa lista irá conter os exemplos de cada classe
            ArrayList<Instance> lista = new ArrayList<>();
            trainingData.put(String.valueOf(iterator.next()), lista);
        }
        
        for (int i = 0; i < D.size(); i++) {
            Set<String> labels = DataSetUtils.getLabelSet(D.get(i)); //Pega os rótulos da instancia
            qtdeRotulos += labels.size();
            ArrayList<Instance> generic; //Lista auxilar usada para receber a lista do hashmap, o exemplo é adicionado a está lista e depois retorna para a HashMap
            for (String label : labels) { //Para cada classe do exemplo
                generic = trainingData.get(label); //busca a lista referente a classe "label"
                if(generic != null){
                    generic.add(D.get(i)); //Adiciona o exemplo à lista
                    trainingData.put(label, generic); //Retorna a lista para a classe
                }
            }
        }
        super.setTrainingData(trainingData);
        this.cardinality = Math.ceil((double)qtdeRotulos/(double)D.size());;
    }
    

    /**
     * @return the model
     */
    public Model getModel() {
        return model;
    }

     
    /**
     * @return the cardinality
     */
    public double getCardinality() {
        return cardinality;
    }

    /**
     * @return the classesConhecidas
     */
    public Set<String> getClassesConhecidas() {
        return classesConhecidas;
    }

    /**
     * @param classesConhecidas the classesConhecidas to set
     */
    public void setClassesConhecidas(Set<String> classesConhecidas) {
        this.classesConhecidas = classesConhecidas;
    }

    /**
     * @param k_ini the k_ini to set
     */
    public void setK_ini(double k_ini) {
        this.k_ini = k_ini;
    }

}
