/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package br;

import NoveltyDetection.KMeansMOAModified;
import NoveltyDetection.MicroCluster;
import java.util.ArrayList;

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
    
}
