/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br;

import NoveltyDetection.ClustreamKernelMOAModified;
import NoveltyDetection.KMeansMOAModified;
import NoveltyDetection.MicroCluster;
import br.ClustreamOfflineBR;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
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
import utils.OnlinePhaseUtils;
import utils.ShortTimeMemory;

/**
 *
 * @author joel
 */
public abstract class OnlinePhase {
    private String algOnl;
    private int timestamp;
    private HashMap<String, ArrayList<MicroCluster>> modelo;
//    private ShortTimeMemory shortTimeMemory;
    private int numExNoveltyDetection;          // minimum number of examples in the unknown memory to execute the ND procedure
    private String outputDirectory;
    private FileWriter fileOn;
    private ArrayList<MicroCluster> sleepMemory;

    public OnlinePhase(HashMap<String, ArrayList<MicroCluster>> modelo, String algOnl, int numExNoveltyDetection, String outputDirectory, FileWriter fileOn) {
        System.out.println("******** Inicio Fase Online ********");
        this.modelo = modelo;
        this.algOnl = algOnl;
        this.numExNoveltyDetection = numExNoveltyDetection;
        this.fileOn = fileOn;
        this.outputDirectory = outputDirectory;
        this.sleepMemory = new ArrayList<>();
    }
    public OnlinePhase(String algOnl, int numExNoveltyDetection, String outputDirectory, FileWriter fileOn) {
        System.out.println("******** Inicio Fase Online ********");
        this.algOnl = algOnl;
        this.numExNoveltyDetection = numExNoveltyDetection;
        this.fileOn = fileOn;
        this.outputDirectory = outputDirectory;
        this.sleepMemory = new ArrayList<>();
    }
    
    
    
    /**
     * Cluster with k-means
     *
     * @param numMClusters
     * @param dataSet
     * @param exampleCluster
     * @return
     * @throws NumberFormatException
     * @throws IOException
     */
    public ArrayList<MicroCluster> createModelKMeansOnline(int numMClusters, ArrayList<Instance> dataSet, int[] exampleCluster) throws NumberFormatException, IOException {
        ArrayList<MicroCluster> modelSet = new ArrayList<>();
        List<ClustreamKernelMOAModified> examples = new LinkedList<>();

        //Adicionando os exemplos ao algoritmo
        for (int k = 0; k < dataSet.size(); k++) {
            double[] data = Arrays.copyOfRange(dataSet.get(k).toDoubleArray(), dataSet.get(k).numOutputAttributes(), dataSet.get(k).numAttributes());
            Instance inst = new DenseInstance(1, data);
            examples.add(new ClustreamKernelMOAModified(inst, dataSet.get(k).numAttributes() - dataSet.get(k).numOutputAttributes(), k));
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
            centrosIni[i] = examples.get(nroaleatorio);
        }

        //execution of the KMeans  
        Clustering centers;
        moa.clusterers.KMeans cm = new moa.clusterers.KMeans();
        centers = cm.kMeans(centrosIni, examples);

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
                model_tmp = new MicroCluster((ClustreamKernelMOAModified) micros.get(w), "", "normal", timestamp);
                modelSet.add(model_tmp);
            }
        }
        return modelSet;
    }

    /**
     * Cluster with clustream
     *
     * @param dataSet exemplos da memória temporária
     * @param numMClusters número de micro-grupos desejado
     * @param cl rótulo
     * @param exampleCluster exemplos que serão removidos da memoria temporaria
     * @return micro-grupos
     * @throws NumberFormatException
     * @throws IOException
     */
//    public ArrayList<MicroCluster> criamodeloCluStreamOnline(ArrayList<Instance> dataSet, int numMClusters, int[] exampleCluster) throws NumberFormatException, IOException {
//        ArrayList<MicroCluster> conjModelos = new ArrayList<MicroCluster>();
//        ClustreamOfflineBR jc = new ClustreamOfflineBR();
//
//        //execute the clustream
//        boolean executeKMeans = false;
//
//        Clustering micros = jc.CluStream(dataSet, this.nLabels, numMClusters, flagMicro, executeKMeans);
//
//        if (micros.size() > 0) {
//            if (flagMicro) {
//                int gruposTemp[] = jc.getClusterExamples();
//                for (int i = 0; i < gruposTemp.length; i++) {
//                    exampleCluster[i] = gruposTemp[i];
//                }
//            } else {
//                int temp[] = jc.getClusteringResults();
//                for (int g = 0; g < jc.getClusteringResults().length; g++) {
//                    exampleCluster[g] = temp[g];
//                }
//            }
//        }
//        for (int w = 0; w < micros.size(); w++) {
//            MicroCluster mdtemp = new MicroCluster((ClustreamKernelMOAModified) micros.get(w), "", "normal", timestamp);
//            // add the temporary model to the decision model 
//            conjModelos.add(mdtemp);
//        }
//        return conjModelos;
//    }
//    
    /**
     * Envia os micro-grupos que não receberam exemplos na última janela para uma memória temporária
     */
//    public void putClusterMemorySleep() throws IOException {
//        for (int i = 0; i < getModelo().size(); i++) {
//            if (getModelo().get(i).getTime() < (timestamp - windowSize)) {
//                SleepModeMem.add(getModelo().get(i));
//                System.out.println("Micro-Grupo Removido: " + i +" classes: " + getModelo().get(i).getLabelClass() + " categoria: " +  getModelo().get(i).getCategory());
//                this.fileOut.write("Micro-Grupo Removido: " + i +" classes: " + getModelo().get(i).getLabelClass() + " categoria: " +  getModelo().get(i).getCategory());
//                this.fileOut.write("\n");
//                getModelo().remove(i);
//                i--;
//            }
//        }
//    }
    
    /**
     * Verifica se a silhueta do grupo é maior que 0 para validar o grupo
     *
     * @param modelUnk novidade
     * @param modeloValidar modelo
     * @return
     */
    public boolean clusterValidationSilhouette(MicroCluster modelUnk, ArrayList<MicroCluster> modeloValidar) {
        double minDistance = Double.MAX_VALUE;
        // calculate the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modeloValidar.size(); i++) {
            double distance = KMeansMOAModified.distance(modeloValidar.get(i).getCenter(), modelUnk.getCenter());
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        double silhouette = (minDistance - modelUnk.getRadius() / 2) / Math.max(minDistance, modelUnk.getRadius() / 2);
        if (silhouette > 0) {
            return true;
        } else {
            return false;
        }
    }
    
    /**
     * Deletes micro-clusters which have not been used for a time period
     * @param windowSize
     * @param modelo
     * @param fileOut
     * @throws IOException 
     */
    public void putClusterMemorySleep(int windowSize, ArrayList<MicroCluster> modelo, FileWriter fileOut) throws IOException {
        ArrayList<MicroCluster> listaMicro = new ArrayList<>();
        this.fileOn.write("Tamanho do Modelo: " + modelo.size());
        for (int i = 0; i < modelo.size(); i++) {
            if (modelo.get(i).getTime() < (timestamp - (windowSize))) {
                listaMicro.add(modelo.get(i));
                getSleepMemory().add(modelo.get(i));
                fileOut.write("Micro-Grupo Removido: " + i +" classes: " + modelo.get(i).getLabelClass() + " categoria: " +  modelo.get(i).getCategory());
                fileOut.write("\n");
                modelo.remove(i);
                i--;
            }
        }
        try{
            fileOn.write("Timestamp: " + this.timestamp + " - Micro grupos removidos: " + listaMicro.size() + " - Tamanho modelo ["+modelo.get(0).getLabelClass()+"]:" + modelo.size() + "\n");
            System.out.println("Timestamp: " + this.timestamp + " - Micro grupos removidos: " + listaMicro.size() + " - Tamanho modelo ["+modelo.get(0).getLabelClass()+"]:" + modelo.size());
        }catch(Exception e){
            
        }
        }
    
    public void plotPeriodicMicroClustersGraph(){
        
    }
    
     /**
     * @return the timestamp
     */
    public int getTimestamp() {
        return timestamp;
    }

    public void incrementarTimeStamp() {
        this.timestamp = this.timestamp + 1;
    }

    /**
     * @return the algOnl
     */
    public String getAlgOnl() {
        return algOnl;
    }

    /**
     * @param algOnl the algOnl to set
     */
    public void setAlgOnl(String algOnl) {
        this.algOnl = algOnl;
    }

    /**
     * @return the modelo
     */
    public HashMap<String, ArrayList<MicroCluster>> getModelo() {
        return modelo;
    }

    /**
     * @param modelo the modelo to set
     */
    public void setModelo(HashMap<String, ArrayList<MicroCluster>> modelo) {
        this.modelo = modelo;
    }

    /**
     * @return the shortTimeMemory
     */
//    public ShortTimeMemory getShortTimeMemory() {
//        return shortTimeMemory;
//    }
//
//    /**
//     * @param shortTimeMemory the shortTimeMemory to set
//     */
//    public void setShortTimeMemory(ShortTimeMemory shortTimeMemory) {
//        this.shortTimeMemory = shortTimeMemory;
//    }

    /**
     * @return the numExNoveltyDetection
     */
    public int getNumExNoveltyDetection() {
        return numExNoveltyDetection;
    }

    /**
     * @param numExNoveltyDetection the numExNoveltyDetection to set
     */
    public void setNumExNoveltyDetection(int numExNoveltyDetection) {
        this.numExNoveltyDetection = numExNoveltyDetection;
    }

    /**
     * @return the outputDirectory
     */
    public String getOutputDirectory() {
        return outputDirectory;
    }

    /**
     * @param outputDirectory the outputDirectory to set
     */
    public void setOutputDirectory(String outputDirectory) {
        this.outputDirectory = outputDirectory;
    }

    /**
     * @return the fileOn
     */
    public FileWriter getFileOn() {
        return fileOn;
    }

    /**
     * @param fileOn the fileOn to set
     */
    public void setFileOn(FileWriter fileOn) {
        this.fileOn = fileOn;
    }

    /**
     * @return the sleepMemory
     */
    public ArrayList<MicroCluster> getSleepMemory() {
        return sleepMemory;
    }

}
