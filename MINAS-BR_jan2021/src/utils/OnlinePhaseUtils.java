/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utils;

import NoveltyDetection.ClustreamKernelMOAModified;
import NoveltyDetection.KMeansMOAModified;
import NoveltyDetection.MicroCluster;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import moa.cluster.CFCluster;
import moa.cluster.Clustering;

/**
 *
 * @author joel
 */
public class OnlinePhaseUtils {
    public static ArrayList<MicroCluster> createModelKMeansLeader(ArrayList<Instance> dataSet, int[] exampleCluster, double maxRadius, int timestamp) throws NumberFormatException, IOException {
        ArrayList<MicroCluster> modelSet = new ArrayList<>();
        List<ClustreamKernelMOAModified> examples = new LinkedList<>();
        
        //Adicionando os exemplos ao algoritmo
        for (int k = 0; k < dataSet.size(); k++) {
            double[] data = Arrays.copyOfRange(dataSet.get(k).toDoubleArray(), dataSet.get(k).numOutputAttributes(), dataSet.get(k).numAttributes());
            Instance inst = new DenseInstance(1, data);
            examples.add(new ClustreamKernelMOAModified(inst, inst.numAttributes(), k));
        }

        //********* K-Means ***********************
        //generate initial centers with leader algorithmn
        ArrayList<Integer> centroids = leaderAlgorithm(dataSet, dataSet.get(0).numOutputAttributes(), maxRadius);
        ClustreamKernelMOAModified[] centrosIni = new ClustreamKernelMOAModified[centroids.size()];
        for (int i = 0; i < centroids.size(); i++) {
            centrosIni[i] = examples.get(centroids.get(i));
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
                ArrayList<Instance> aux = new ArrayList<Instance>();
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
        for (int w = 0; w < centroids.size(); w++) {
            if ((micros.get(w) != null)) {
                model_tmp = new MicroCluster((ClustreamKernelMOAModified) micros.get(w), "", "ext", timestamp);
                modelSet.add(model_tmp);
            }
        }
        return modelSet;
    }
    
    public static HashMap<String, ArrayList<MicroCluster>> KMeansLeader(Set<MicroCluster> dataSet, double maxD) throws NumberFormatException, IOException {
        List<ClustreamKernelMOAModified> examples = new LinkedList<>();
        
        //Adicionando os exemplos ao algoritmo
        int cont = 0;
        ArrayList<MicroCluster> listAux = new ArrayList<MicroCluster>();
        for (Iterator<MicroCluster> iterator = dataSet.iterator(); iterator.hasNext();) {
            MicroCluster next = iterator.next();
            listAux.add(next);
            double[] data = next.getCenter();
            Instance inst = new DenseInstance(1, data);
            examples.add(new ClustreamKernelMOAModified(inst, inst.numAttributes(), cont));
            cont++;
        }

        //********* K-Means ***********************
        //generate initial centers with leader algorithmn
        ArrayList<Integer> centroids = leaderAlgorithm(listAux,  maxD);
        ClustreamKernelMOAModified[] centrosIni = new ClustreamKernelMOAModified[centroids.size()];
        HashMap<String, ArrayList<MicroCluster>> retorno = new HashMap<>();
        for (int i = 0; i < centroids.size(); i++) {
            centrosIni[i] = examples.get(centroids.get(i));
            retorno.put(""+i, new ArrayList<MicroCluster>());
        }

        //execution of the KMeans  
        Clustering centers;
        moa.clusterers.KMeans cm = new moa.clusterers.KMeans();
        centers = cm.kMeans(centrosIni, examples);
        
        //*********results     
        // transform the results of kmeans in a data structure used by MINAS
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
            ArrayList<MicroCluster> res = retorno.get(""+closestCluster);
            res.add(listAux.get(j));
        }

        return retorno;
    }
    
    /**
     * Get the kmeans k number
     * @param dataSet
     * @return 
     */
    private static ArrayList<Integer> leaderAlgorithm(ArrayList<Instance> dataSet, int qtdeTotalClasses, double maxRadius) {
        ArrayList<Integer> centroids = new ArrayList<Integer>();
        centroids.add(0);
        for (int i = 1; i < dataSet.size(); i++) {
            boolean centroid = false;
            for (int j = 0; j < centroids.size(); j++) {
                double[] data1 = Arrays.copyOfRange(dataSet.get(i).toDoubleArray(), qtdeTotalClasses, dataSet.get(1).numAttributes());
                double[] data2 = Arrays.copyOfRange(dataSet.get(centroids.get(j)).toDoubleArray(), qtdeTotalClasses, dataSet.get(centroids.get(j)).numAttributes());
                double dist = KMeansMOAModified.distance(data1, data2);
                if(dist < maxRadius){
                    centroid = false;
                    break;
                }else{
                    centroid = true;
                }
            }
            if(centroid){
                centroids.add(i);
            }
        }
        return centroids;
    }
    
    /**
     * Get the kmeans k number
     * @param dataSet
     * @return 
     */
    private static ArrayList<Integer> leaderAlgorithm(ArrayList<MicroCluster> dataSet, double maxRadius) {
        ArrayList<Integer> centroids = new ArrayList<Integer>();
        centroids.add(0);
        for (int i = 1; i < dataSet.size(); i++) {
            boolean centroid = false;
            for (int j = 0; j < centroids.size(); j++) {
                double[] data1 = dataSet.get(i).getCenter();
                double[] data2 = dataSet.get(centroids.get(j)).getCenter();
                double dist = KMeansMOAModified.distance(data1, data2);
                if(dist < maxRadius){
                    centroid = false;
                    break;
                }else{
                    centroid = true;
                }
            }
            if(centroid){
                centroids.add(i);
            }
        }
        return centroids;
    }
    
    
    
    /**
     * Get the greatest micro-cluster radius of the model
     * @param modelo
     * @return 
     */
    public static double getMaxRadius(ArrayList<MicroCluster> modelo) {
        double maxRadius = 0;
        for (MicroCluster m : modelo) {
            if(m.getRadius() > maxRadius)
                maxRadius = m.getRadius();
        }
        return maxRadius;
    }
    
    /**
     * Get the greatest micro-cluster radius of the model
     * @param modelo
     * @return 
     */
    public static double getMaxRadius(HashMap<String, ArrayList<MicroCluster>> modelo) {
        double maxRadius = 0;
        for (Map.Entry<String, ArrayList<MicroCluster>> entry : modelo.entrySet()) {
            ArrayList<MicroCluster> value = entry.getValue();
            if(maxRadius < getMaxRadius(value)){
                maxRadius = getMaxRadius(value);
            }
        }
        return maxRadius;
    }
}
