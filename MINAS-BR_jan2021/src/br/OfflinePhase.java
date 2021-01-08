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
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import moa.cluster.CFCluster;
import moa.cluster.Clustering;

/**
 *
 * @author joel
 */
public abstract class OfflinePhase {
    private HashMap<String, ArrayList<Instance>> trainingData;
    private String algOff;                                      //algoritmo de agrupamento na fase offline  
    private FileWriter fileOut;
    private String directory;

    public OfflinePhase(String algOff, FileWriter fileOut, String directory) {
        System.out.println("******** Inicio Fase Offline ********");
        this.algOff = algOff;
        this.fileOut = fileOut;
        this.directory = directory;
    }
    
    public ArrayList<MicroCluster> createModelKMeansOffline(ArrayList<Instance> dataSet, String label, int[] exampleCluster, int numMClusters) throws NumberFormatException, IOException {
        ArrayList<MicroCluster> modelSet = new ArrayList<MicroCluster>();
        List<ClustreamKernelMOAModified> examples = new LinkedList<ClustreamKernelMOAModified>();
        if(numMClusters < 1){
            numMClusters = 1;
        }
        int indexLabels = dataSet.get(0).numOutputAttributes();
        int numAtt =  dataSet.get(0).numAttributes() - indexLabels;
        
        
        //************read dataset *************************
        //read examples from the file to the memory to execute Kmeans
        for (int k = 0; k < dataSet.size(); k++) {
//            System.out.println(Arrays.toString(dataSet.get(k).toDoubleArray()));
            double[] data = Arrays.copyOfRange(dataSet.get(k).toDoubleArray(), dataSet.get(k).numOutputAttributes(), dataSet.get(k).numAttributes());
            Instance inst = new DenseInstance(1, data);
            examples.add(new ClustreamKernelMOAModified(inst, numAtt, 0));
            exampleCluster = new int[dataSet.size()];
        }

        //********* K-Means ***********************
        //generate initial centers aleatory
        ClustreamKernelMOAModified[] centrosIni = new ClustreamKernelMOAModified[numMClusters];
        int nroaleatorio;
        List<Integer> numeros = new ArrayList<Integer>();
        for (int c = 0; c < examples.size(); c++) {
            numeros.add(c);
        }
        Collections.shuffle(numeros);
        for (int i = 0; i < numMClusters; i++) {
            nroaleatorio = numeros.get(i).intValue();
            centrosIni[i] = examples.get(i/* nroaleatorio*/);
        }

        //execution of the KMeans  
        Clustering centers;
        moa.clusterers.KMeans cm = new moa.clusterers.KMeans();
//        try{
            centers = cm.kMeans(centrosIni, examples);
//        }catch(Exception e){
//            System.out.println("");
//        }
        
        //*********results     
        // transform the results of kmeans in a data structure used by MINAS
        CFCluster[] res = new CFCluster[centers.size()];
        for (int j = 0; j < examples.size(); j++) {
            // Find closest kMeans cluster
            double minDistance = Double.MAX_VALUE;
            int closestCluster = 0;
            for (int i = 0; i < centers.size(); i++) {
                double distance = KMeansMOAModified.distance(centers.get(i).getCenter(), examples.get(j).getCenter());
                if (distance < minDistance) {
                    closestCluster = i;
                    minDistance = distance;
                }
            }

            // add to the cluster
            if (res[closestCluster] == null) {
                res[closestCluster] = (CFCluster) examples.get(j).copy();
            } else {
                res[closestCluster].add(examples.get(j));
            }
            exampleCluster[j] = closestCluster;
        }

        Clustering micros;
        micros = new Clustering(res);

        //*********remove micro-cluster with few examples
        ArrayList<ArrayList<Integer>> mapClustersExamples = new ArrayList<ArrayList<Integer>>();
        for (int a = 0; a < centrosIni.length; a++) {
            mapClustersExamples.add(new ArrayList<Integer>());
        }
        for (int g = 0; g < exampleCluster.length; g++) {
            mapClustersExamples.get(exampleCluster[g]).add(g);
        }

        int value;
        for (int i = 0; i < micros.size(); i++) {
            //remove micro-cluster with less than 3 examples
            if (micros.get(i) != null) {
                if (((ClustreamKernelMOAModified) micros.get(i)).getWeight() < 3) {
                    value = -1;
                } else {
                    value = i;
                }

                for (int j = 0; j < mapClustersExamples.get(i).size(); j++) {
                    exampleCluster[mapClustersExamples.get(i).get(j)] = value;
                }
                if (((ClustreamKernelMOAModified) micros.get(i)).getWeight() < 3) {
                    micros.remove(i);
                    mapClustersExamples.remove(i);
                    i--;
                }
            } else {
                micros.remove(i);
                mapClustersExamples.remove(i);
                i--;
            }
        }

        MicroCluster model_tmp;
        for (int w = 0; w < numMClusters; w++) {
            if ((micros.get(w) != null)) {
                model_tmp = new MicroCluster((ClustreamKernelMOAModified) micros.get(w), label, "normal", 0);
                modelSet.add(model_tmp);
            }
        }
        return modelSet;
    }
    
    /**
     * Cria modelo de decisão através do algoritmo CluStream
     * @param examples of class
     * @param label of class
     * @param numMClusters
     * @return a list of microclusters that represents a class
     * @throws NumberFormatException
     * @throws IOException 
     */
    public ArrayList<MicroCluster> criarmodeloCluStreamOffline(ArrayList<Instance> examples, String label, int numMClusters) throws NumberFormatException, IOException {
        ArrayList<MicroCluster> conjModelos = new ArrayList<>();
        ClustreamOfflineBR jc = new ClustreamOfflineBR();

        Clustering micros = jc.CluStream(examples, examples.get(0).numOutputAttributes(), numMClusters, true, true /*executa kmeans*/);

        for (int w = 0; w < micros.size(); w++) {
            MicroCluster mdtemp = new MicroCluster((ClustreamKernelMOAModified) micros.get(w), label, "normal", 0);
            // add the temporary model to the decision model 
            conjModelos.add(mdtemp);
        }
        return conjModelos;
    }

    /**
     * @return the trainingData
     */
    public HashMap<String, ArrayList<Instance>> getTrainingData() {
        return trainingData;
    }

    /**
     * @param trainingData the trainingData to set
     */
    public void setTrainingData(HashMap<String, ArrayList<Instance>> trainingData) {
        this.trainingData = trainingData;
    }

    /**
     * @return the algOff
     */
    public String getAlgOff() {
        return algOff;
    }

    /**
     * @return the fileOut
     */
    public FileWriter getFileOut() {
        return fileOut;
    }

    /**
     * @return the directory
     */
    public String getDirectory() {
        return directory;
    }

    /**
     * @param directory the directory to set
     */
    public void setDirectory(String directory) {
        this.directory = directory;
    }
    
    

    
    
}
