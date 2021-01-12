/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br;

import NoveltyDetection.KMeansMOAModified;
import NoveltyDetection.MicroCluster;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 *
 * @author joel
 */
public class MicroClusterBR {
    private double threshold;
    private double averOut;
    private MicroCluster microCluster;

    public MicroClusterBR(MicroCluster microCluster) {
        this.microCluster = microCluster;
    }

    /**
     * @return the threshold
     */
    public double getThreshold() {
        return threshold;
    }

    /**
     * @return the averOut
     */
    public double getAverOut() {
        
        return averOut;
    }

    /**
     * @return the microCluster
     */
    public MicroCluster getMicroCluster() {
        return microCluster;
    }
    
    /**
     * Silhouette validation
     *
     * @param modeloValidar model
     * @return
     */
    public boolean clusterValidationSilhouette(ArrayList<MicroClusterBR> modeloValidar) {
        double minDistance = Double.MAX_VALUE;
        // calculate the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modeloValidar.size(); i++) {
            double distance = KMeansMOAModified.distance(modeloValidar.get(i).getMicroCluster().getCenter(), this.microCluster.getCenter());
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        double silhouette = (minDistance - this.microCluster.getRadius() / 2) / Math.max(minDistance, this.microCluster.getRadius() / 2);
        if (silhouette > 0) {
            return true;
        } else {
            return false;
        }
    }

    public void calculateInitialAverOutput(ArrayList<double[]> X_b) {
        double sum = 0;
        for (double[] x_i : X_b) {
            sum += Math.exp(-KMeansMOAModified.distance(this.getMicroCluster().getCenter(), x_i));
        }
        this.averOut = sum / X_b.size();
    }

    public void calculateInicialThreshold(HashMap<String, Integer> mtxLabelsFrequencies) {
        String j = this.getMicroCluster().getLabelClass();
        double p_yj = mtxLabelsFrequencies.get(j + "," + j);
        double prod = 1;
        for (Map.Entry<String, Integer> entry : mtxLabelsFrequencies.entrySet()) {
            String key[] = entry.getKey().split(",");
            if(key[1].equals(j)){
                double p_yk_yj = entry.getValue() / p_yj;
                prod *= p_yk_yj * this.averOut;
            }
        }
        this.threshold = p_yj * prod;
    }
    
}
