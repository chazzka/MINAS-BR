package main;


import br.Model;
import br.OfflinePhase;
import br.OnlinePhase;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Prediction;
import dataSource.DataSetUtils;
import static dataSource.DataSetUtils.slipTrainTest;
import dataSource.LabelSetMining;
import static dataSource.LabelSetMining.removeInfrequentLabels;
import evaluate.Evaluator;
import evaluate.EvaluatorBR;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import meka.classifiers.multilabel.incremental.BRUpdateable;
import meka.classifiers.multilabel.incremental.CCUpdateable;
import meka.classifiers.multilabel.incremental.PSUpdateable;
import moa.classifiers.multilabel.MEKAClassifier;
import moa.classifiers.multilabel.MultilabelHoeffdingTree;
import moa.classifiers.multilabel.meta.OzaBagAdwinML;
import moa.classifiers.multilabel.trees.ISOUPTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstanceExample;
import moa.options.WEKAClassOption;
import moa.streams.MultiTargetArffFileStream;
import utils.FilesOutput;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instances;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 * Classe principal utilizada para comparação entre diferentes configurações de
 * parâmetros do MINAS-LP
 *
 * @author joel
 */
public class Main {

    public static void main(String[] args) throws Exception {
          
        //****************General*********************
//        String dataSetName = args[1];
//        String trainPath = args[2];
//        String testPath = args[3];
//        int L = Integer.parseInt(args[4]);
//        String outputDirecotory = "";
//        if(args[0].equals("-p")){
//            outputDirecotory = args[5];
//            
//            experimentsParameters(dataSetName,
//                trainPath,
//                testPath,
//                L,
//                outputDirecotory);
//        }else{
//            double k_ini = Double.parseDouble(args[5]);
//            String theta = args[6];
//            String omega = args[7];
//            outputDirecotory = args[8] + "/" + dataSetName;
//            
//            experimentsMethods(trainPath, 
//                testPath, 
//                outputDirecotory,
//                L, 
//                k_ini,
//                theta, 
//                omega,
//                "1.1",
//                "kmeans+leader",
//                "JI");
//        }
//        //****************MOA-3C*********************
        String dataSetName = "MOA-3C_resetingMtxT";
        String trainPath = "/home/joel/datasets/datasets_sinteticos/MOA-3C-5C-2D/MOA-3C-5C-2D-train.arff";
        String testPath = "/home/joel/datasets/datasets_sinteticos/MOA-3C-5C-2D/MOA-3C-5C-2D-test.arff";
        String outputDirecotory = "/home/joel/MINAS-BR_ASOC/results_fev/"+dataSetName+"/";
        double k_ini = 0.01;
        String theta = "1000";
        String omega = "2000";
        int L = 5;
        //*****************************************
        
//        //****************MOA1*********************
//        String dataSetName = "MOA1";
////        String dataSetName = "MOA1_kini=0.001";
//        String trainPath = "/home/joel/Documents/datasets/datasets_sinteticos/MOA-5C-7C-2D/MOA-5C-7C-2D-train.arff";
//        String testPath = "/home/joel/Documents/datasets/datasets_sinteticos/MOA-5C-7C-2D/MOA-5C-7C-2D-test.arff";
//        String outputDirecotory = "results_jan2021/"+dataSetName+"/";
//        double k_ini = 0.01;
////        double k_ini = 0.001;
//        String theta = "1000";
//        String omega = "2000";
//        int L = 7;
//        //*****************************************

////        //****************MOA2*********************
//        String dataSetName = "MOA2";
//        String trainPath = "/home/joel/Documents/datasets/datasets_sinteticos/4CRE-V2/4CRE-V2-train.arff";
//        String testPath = "/home/joel/Documents/datasets/datasets_sinteticos/4CRE-V2/4CRE-V2-test.arff";
//        String outputDirecotory = "results_jan2021/"+dataSetName+"/";
//        double k_ini = 0.01;
//        String theta = "200";
//        String omega = "200";
//        int L = 4;
//        //*****************************************
        
        experimentsMethods(trainPath, 
                testPath, 
                outputDirecotory,
                L, 
                k_ini,
                theta, 
                omega,
                "1",
                "kmeans+leader",
                "JI");
        
//        experimentsParameters(dataSetName,
//                trainPath,
//                testPath,
//                L,
//                outputDirecotory);
    }
    
    public static void convertArffFile(String train, String test, String dataSetName) throws Exception{
        Instances D_train = DataSetUtils.dataFileToInstance(train);
        Instances D_test = DataSetUtils.dataFileToInstance(test);
        int L = D_test.classIndex();
        int numAtt = (D_train.instance(0).numAttributes() - L);
        String fileName = train.replace("-train.arff", ".arff");
        FileWriter dataSetFile = new FileWriter(new File(fileName), false);
        //write the file's header
        dataSetFile.write("@relation '" + dataSetName + ": -C "+ L+ "'\n");
        dataSetFile.write("\n");
        
        for (int i = 0; i < L; i++) {
            dataSetFile.write("@attribute class"+i+" {0, 1}\n");
        }
        for (int i = 1; i < numAtt; i++) {
            dataSetFile.write("@attribute att"+i+"\n");
        }
        dataSetFile.write("\n");
        dataSetFile.write("@data\n");
        
        for (int j = 0; j < D_train.numInstances(); j++) {
            for (int i = 0; i < L; i++) {
                double value = D_train.instance(j).value(i);
                dataSetFile.write((int)value+",");
            }
            for (int i = 1; i < numAtt; i++) {
                double value = D_train.instance(j).value(L+i);
                if(i == numAtt-1 ){
                    dataSetFile.write(""+value);
                }else{
                    dataSetFile.write(value+",");
                }
            }
            dataSetFile.write("\n");
        }
        
        for (int j = 0; j < D_test.numInstances(); j++) {
            for (int i = 0; i < L; i++) {
                double value = D_test.instance(j).value(i);
                dataSetFile.write((int)value+",");
            }
            for (int i = 1; i < numAtt; i++) {
                double value = D_test.instance(j).value(L+i);
                if(i == numAtt-1 ){
                    dataSetFile.write(""+value);
                }else{
                    dataSetFile.write(value+",");
                }
            }
            dataSetFile.write("\n");
        }
        dataSetFile.close();
    }
    
    private static void arffToCsv(String dataSetPath) throws Exception{
        Instances D_ = DataSetUtils.dataFileToInstance(dataSetPath);
        int L = D_.classIndex();

        MultiTargetArffFileStream stream = new MultiTargetArffFileStream(dataSetPath, String.valueOf(L));
        stream.prepareForUse();
        
        String fileName = dataSetPath.replace("arff", "csv");
        FileWriter csvFile = new FileWriter(new File(fileName), false);
        csvFile.write("Class");
        Instance begin = stream.nextInstance().instance;
        for (int i = 0; i < begin.numInputAttributes(); i++) {
            csvFile.write(";"+begin.inputAttribute(i).name());
        }
        csvFile.write("\n");
        stream.restart();
        while(stream.hasMoreInstances()) {
            begin = stream.nextInstance().instance;
            csvFile.write(DataSetUtils.getLabelSet(begin).toString());
            for (int i = 0; i < begin.numInputAttributes(); i++) {
                double y = begin.valueInputAttribute(i);
                csvFile.write(";"+begin.valueInputAttribute(i));
            }
            csvFile.write("\n");
        }
        csvFile.close();
    }
    
    private static void experimentsParameters(String dataSetName,
            String trainPath, 
            String testPath,
            int L,
            String outputDirectory) throws IOException, Exception {
        
        ArrayList<Instance> train = new ArrayList<Instance>();
        ArrayList<Instance> test = new ArrayList<Instance>();
        
        MultiTargetArffFileStream file = new MultiTargetArffFileStream(trainPath, String.valueOf(L));
        file.prepareForUse();
        while(file.hasMoreInstances()){
            train.add(file.nextInstance().getData());
        }
        file.restart();
        
        file = new MultiTargetArffFileStream(testPath, String.valueOf(L));
        file.prepareForUse();
        
        while(file.hasMoreInstances()){
            test.add(file.nextInstance().getData());
        }
        file.restart();
//        double[] theta = {0.1,0.75};
//        int[] omega = {2000};
//        double[] f = {1.1};
//        double[] k_ini = {0.001};
        double[] theta = {0.1,0.25,0.5,0.75,1};
        int[] omega = {200, 500, 1000, 2000};
        double[] f = {0.5, 0.75, 1.1, 1.3};
        double[] k_ini = {0.01, 0.05, 0.1, 0.25};

        outputDirectory = outputDirectory + "/parameters_sensitivity/"+dataSetName+"/";
        FilesOutput.createDirectory(outputDirectory);
        
        
        FileWriter F1M = new FileWriter(new File(outputDirectory + "/F1M.csv"), false);
        FileWriter pr = new FileWriter(new File(outputDirectory + "/precision.csv"), false);
        FileWriter re = new FileWriter(new File(outputDirectory + "/recall.csv"), false);
        FileWriter SA = new FileWriter(new File(outputDirectory + "/SA.csv"), false);
        
        F1M.write("theta-omega"+ ",");
        pr.write("theta-omega"+ ",");
        re.write("theta-omega"+ ",");
        SA.write("theta-omega"+ ",");
        
        for (int i = 0; i < f.length; i++) {
            for (int j = 0; j< k_ini.length ; j++) {
                F1M.write("f_"+f[i]+"_k_"+k_ini[j] + ",NPs,");
                pr.write("f_"+f[i]+"_k_"+k_ini[j] + ",");
                re.write("f_"+f[i]+"_k_"+k_ini[j] + ",");
                SA.write("f_"+f[i]+"_k_"+k_ini[j] + ",");
            }
            
        }
        
        F1M.write("\n");
        pr.write("\n");
        re.write("\n");
        SA.write("\n");
        String dir = outputDirectory;
        
        for (int i = 0; i < theta.length; i++) {
            for (int j = 0; j < omega.length; j++) {
                if(theta[i] <= omega[j]){
                    F1M.write(theta[i] + "-" + omega[j] + ",");
                    pr.write(theta[i] + "-" + omega[j]+ ",");
                    re.write(theta[i] + "-" + omega[j]+ ",");
                    SA.write(theta[i] + "-" + omega[j]+ ",");
                    for (int k = 0; k < f.length; k++) {
                        for (int l = 0; l < k_ini.length; l++) {
                            outputDirectory = dir + theta[i] + "_" + omega[j] + "_" + f[k]+ "_" + k_ini[l] +"/";
                            System.out.println("***********"+theta[i] + "_" + omega[j] + "_" + f[k]+ "_" + k_ini[l]+"******************");
                            FilesOutput.createDirectory(outputDirectory);
                            EvaluatorBR avMINAS = MINAS_BR(train,
                                    test,
                                    L, 
                                    k_ini[l],
                                    (int) theta[i]*omega[j],
                                    omega[j], 
                                    f[k],
                                    outputDirectory);
                            F1M.write(avMINAS.getAvgF1M() + "," + avMINAS.getQtdeNP() + ",");
                            pr.write(avMINAS.getAvgPr()+ ",");
                            re.write(avMINAS.getAvgRe() + ",");
                            SA.write(avMINAS.getAvgSA() + ",");
                        }
                    }
                    F1M.write("\n");
                    pr.write("\n");
                    re.write("\n");
                    SA.write("\n");
                }
            }
        }
        F1M.close();
        pr.close();
        re.close();
        SA.close();
    }
    
    public static EvaluatorBR MINAS_BR(ArrayList<Instance> train,
            ArrayList<Instance> test,
            int L, 
            double k_ini, 
            int theta, 
            int omega, 
            double f, 
            String outputDirectory) throws IOException, Exception {
        
//        Instances D_ = DataSetUtils.dataFileToInstance(dataSetPath);
//        int L = D_.classIndex();
//        int streamSize = D_.numInstances();
//        int L = 81;
//        int streamSize = 269648;
        
//        MultiTargetArffFileStream stream = new MultiTargetArffFileStream(dataSetPath, String.valueOf(L));
//        stream.prepareForUse();
//        ArrayList<Instance> train = new ArrayList<Instance>();
//        ArrayList<Instance> test = new ArrayList<Instance>();
//        slipTrainTest(train, test, stream, streamSize, 0.1);
//        int[] dist = DataSetUtils.getLabelsDistribution(train);
//        int[] distTest = DataSetUtils.getLabelsDistribution(train);
//        float[] windowsCardinalities = DataSetUtils.getWindowsCardinalities(test, evaluationWindowSize, L);

        //Create output files
        FileWriter filePredictions = new FileWriter(new File(outputDirectory + "/predictionsInfo.csv"), false); //Armazena informações da fase online
        filePredictions.write("timestamp;actual;predicted" + "\n");
        FileWriter fileOff = new FileWriter(new File(outputDirectory + "/faseOfflineInfo.txt"), false); //Armazena informações da fase online
        FileWriter fileOut = new FileWriter(new File(outputDirectory + "/results.txt"), false); //Armazena informações da fase de treinamento
        
        int evaluationWindowSize = (int) Math.ceil(test.size()/50);
        
        OfflinePhase treino = new OfflinePhase(train, k_ini, fileOff, outputDirectory);
        Model model = treino.getModel();
        model.setEvaluationWindowSize(evaluationWindowSize);
        model.writeCurrentCardinality(1, outputDirectory);
        
        fileOff.write("Known Classes: " + model.getAllLabel().size() + "\n");
         fileOff.write("Train label cardinality: " + model.getCurrentCardinality() + "\n");
//        fileOff.write("Windows label cardinality: " + Arrays.toString(windowsCardinalities) + "\n");
        fileOff.write("Number of examples: " + (train.size()+test.size()) + "\n");
        fileOff.write("Number of attributes: " + train.get(0).numInputAttributes() +"\n");
        
        EvaluatorBR av = new EvaluatorBR(L, model.getModel().keySet(), "MINAS-BR"); 
        OnlinePhase onlinePhase = new OnlinePhase(theta, f, outputDirectory, fileOut, "kmeans+leader");
        
        //Classification phase
        for (int i = 0; i < test.size(); i++) {
            onlinePhase.incrementarTimeStamp();
            System.out.println("Timestamp: " + onlinePhase.getTimestamp());
            onlinePhase.classify(model, av, test.get(i),filePredictions);
            
            //for each model deletes the micro-clusters wich have not been used
            if((onlinePhase.getTimestamp()%omega) == 0){
                model.resetMtxLabelFrequencies(omega);
                model.clearSortTimeMemory(omega, onlinePhase.getTimestamp(),fileOut, false);
                onlinePhase.removeOldMicroClusters(omega, model, fileOut);
            }
            if((onlinePhase.getTimestamp()%evaluationWindowSize) == 0){
                model.writeBayesRulesElements(onlinePhase.getTimestamp(), outputDirectory);
                model.writeCurrentCardinality(onlinePhase.getTimestamp(), outputDirectory);
                model.associatesNPs(evaluationWindowSize, onlinePhase.getTimestamp(), "JI");
//                av.getDeletedExamples().add(model.getShortTimeMemory().getQtdeExDeleted());
                av.updateExampleBasedMeasure(model, evaluationWindowSize);
                av.updateLabelBasedMeasure(model, evaluationWindowSize);
            }
            if(i == test.size()-1 && (onlinePhase.getTimestamp()%evaluationWindowSize)>0){
                model.associatesNPs(evaluationWindowSize, onlinePhase.getTimestamp(), "JI");
                av.updateExampleBasedMeasure(model, evaluationWindowSize);
                av.updateLabelBasedMeasure(model, evaluationWindowSize);
            }
        }
        onlinePhase.getExtInfo().close();
        av.setQtdeNP(model.getNPs().size());
        av.writeMeasuresOverTime(outputDirectory);
        av.writeConceptEvolutionNP(model, outputDirectory);
        fileOut.close();
        filePredictions.close();
        fileOff.close();
        System.out.println("Number of examples sent to short-time-memory = " + onlinePhase.getExShortTimeMem());
        model.getPnInfo().write("Number of examples sent to short-time-memory = " + onlinePhase.getExShortTimeMem() + "\n");
        System.out.println("Number of examples removed from short-time-memory = " + model.getShortTimeMemory().getQtdeExDeleted());
        model.getPnInfo().write("Number of examples removed from short-time-memory = " + model.getShortTimeMemory().getQtdeExDeleted()+ "\n");
        System.out.println("Number of NPs = " + model.getNPs().size());
        model.getPnInfo().write("Number of NPs = " + model.getNPs().size()+ "\n");
        model.getPnInfo().close();
        return av;
    }
    
    

    /**
     * Experiments to compare MINAS-BR with others methods
     *
     * @param dataSetName
     * @param dataSetPath
     * @param omega window size
     * @param theta limit of short-term memory 
     * @param f
     * @param algOn
     * @param evMetric
     * @param outputDirectory
     * @throws Exception
     */
    public static void experimentsMethods(String trainPath, 
            String testPath,
            String outputDirectory,
            int L, 
            double k_ini,
            String theta,
            String omega, 
            String f,
            String algOn,
            String evMetric) throws Exception {
        
       FilesOutput.createDirectory(outputDirectory);
       ArrayList<Instance> train = new ArrayList<Instance>();
       ArrayList<Instance> test = new ArrayList<Instance>();
        
        MultiTargetArffFileStream file = new MultiTargetArffFileStream(trainPath, String.valueOf(L));
        file.prepareForUse();
        while(file.hasMoreInstances()){
            train.add(file.nextInstance().getData());
        }
        file.restart();
        
        file = new MultiTargetArffFileStream(testPath, String.valueOf(L));
        file.prepareForUse();
        
        while(file.hasMoreInstances()){
            test.add(file.nextInstance().getData());
        }
        file.restart();
        
        ArrayList<Instance> aux = new ArrayList<>();
        aux.addAll(train);
        aux.addAll(test);
        float cardinalityTrain = DataSetUtils.getCardinality(train, L);
        float labelCardinality = DataSetUtils.getCardinality(aux, L);
        FileWriter DsInfos = new FileWriter(new File(outputDirectory + "/dataSetInfo.txt"), false);
        

        DsInfos.write("Train label cardinality: " + cardinalityTrain + "\n");
        DsInfos.write("General label cardinality: " + labelCardinality + "\n");
//        DsInfos.write("Windows label cardinality: " + Arrays.toString(windowsCardinalities) + "\n");
        DsInfos.write("Number of examples: " + train.size()+test.size() + "\n");
        DsInfos.write("Number of attributes: " + train.get(0).numInputAttributes() +"\n");
        DsInfos.close();
        
        ArrayList<Evaluator> av = new ArrayList<Evaluator>();
        av.add(MINAS_BR(train,
                test,
                L, 
                k_ini,
                Integer.valueOf(theta), 
                Integer.valueOf(omega), 
                Double.valueOf(f),
                outputDirectory)
        );

        //EaHTps
//        OzaBagAdwinML EaHTps = new OzaBagAdwinML();
//        MultilabelHoeffdingTree ht = new MultilabelHoeffdingTree();
//         ht.setModelContext(file.getHeader());
//         ht.prepareForUse();
//         ht.resetLearningImpl();
//        MEKAClassifier ps = new MEKAClassifier();
//        PSUpdateable pse = new PSUpdateable();
////        pse.setClassifier(new weka.classifiers.trees.HoeffdingTree());
//        ps.baseLearnerOption.setCurrentObject(pse);
//        ps.setModelContext(file.getHeader());
//        ps.prepareForUse();
//        ps.resetLearningImpl();
////        ht.learnerOption.setCurrentObject(ps);
//        EaHTps.baseLearnerOption.setCurrentObject(ht);
////        ps.setModelContext(file.getHeader());
//        EaHTps.setModelContext(file.getHeader());
//        EaHTps.prepareForUse();
//        EaHTps.resetLearningImpl();
//        Evaluator avEaHTps = new Evaluator(m, C_con, "EaMLHT");
        
//        OzaBagAdwinML EaHTps = new OzaBagAdwinML();
//        MultilabelHoeffdingTree ht = new MultilabelHoeffdingTree();
//        MEKAClassifier ps = new MEKAClassifier();
//        PSUpdateable pse = new PSUpdateable();
////        pse.setClassifier(new NaiveBayesUpdateable());
//        ps.baseLearnerOption.setCurrentObject(pse); 
//        ht.learnerOption.setCurrentObject(ps);
//        EaHTps.baseLearnerOption.setCurrentObject(ht);
//        EaHTps.setModelContext(file.getHeader());
//        EaHTps.prepareForUse();
//        EaHTps.resetLearningImpl();
//        Evaluator avEaHTps = new Evaluator(m, C_con, "EaMLHT");
//        
//        //EaBR
//        OzaBagAdwinML EaBR = new OzaBagAdwinML();
//        MEKAClassifier BRe = new MEKAClassifier();
//        BRUpdateable brUpdateable = new BRUpdateable();
////        brUpdateable.setClassifier(new NaiveBayesUpdateable());
//        BRe.baseLearnerOption.setCurrentObject(brUpdateable);
//        BRe.setModelContext(file.getHeader());
//        BRe.prepareForUse();
//        BRe.resetLearningImpl();
//        EaBR.baseLearnerOption.setCurrentObject(BRe);
//        EaBR.setModelContext(file.getHeader());
//        EaBR.prepareForUse();
//        EaBR.resetLearningImpl();
//        Evaluator avEaBR = new Evaluator(m, C_con, "EaBR");
//        
//        //EaCC
//        OzaBagAdwinML EaCC = new OzaBagAdwinML();
//        MEKAClassifier CCe = new MEKAClassifier();
//        CCUpdateable ccUpdateable = new CCUpdateable();
////        ccUpdateable.setClassifier(new NaiveBayesUpdateable());
//        CCe.baseLearnerOption.setCurrentObject(ccUpdateable);
//        CCe.setModelContext(file.getHeader());
//        CCe.prepareForUse();
//        CCe.resetLearningImpl();
//        EaCC.baseLearnerOption.setCurrentObject(CCe);
//        EaCC.setModelContext(file.getHeader());
//        EaCC.prepareForUse();
//        EaCC.resetLearningImpl();
//        Evaluator avEaCC = new Evaluator(m, C_con, "EaCC");
//        
//        //CC do moa sem feedback
//        MEKAClassifier CC = new MEKAClassifier();
//        CCUpdateable ccU = new CCUpdateable();
////        ccU.setClassifier(new NaiveBayesUpdateable());
//        CC.baseLearnerOption.setCurrentObject(ccU);
//        CC.setModelContext(file.getHeader());
//        CC.prepareForUse();
//        CC.resetLearningImpl();
//        Evaluator avCC = new Evaluator(m, C_con, "CC");
//        
//        //PS do moa sem feedback
//        MEKAClassifier PS = new MEKAClassifier();
//        PSUpdateable psUpdateable = new PSUpdateable();
////        psUpdateable.setClassifier(new NaiveBayesUpdateable());
//        PS.baseLearnerOption.setCurrentObject(psUpdateable);
//        PS.setModelContext(file.getHeader());
//        PS.prepareForUse();
//        PS.resetLearningImpl();
//        Evaluator avPS = new Evaluator(m, C_con, "PS");
//        
//        //MLHT without feedback
//        MultilabelHoeffdingTree MLHT = new MultilabelHoeffdingTree();
//        MEKAClassifier PSHT = new MEKAClassifier();
//        PSUpdateable psHT = new PSUpdateable();
////        psHT.setClassifier(new NaiveBayesUpdateable());
//        PSHT.baseLearnerOption.setCurrentObject(psHT);
//        MLHT.learnerOption.setCurrentObject(PSHT);
//        MLHT.setModelContext(file.getHeader());
//        MLHT.prepareForUse();
//        MLHT.resetLearningImpl();
//        Evaluator avMLHT = new Evaluator(m, C_con, "MLHT");
//        
//        //ISOUTree without feedback
//        ISOUPTree iTree = new ISOUPTree();
//        iTree.setModelContext(file.getHeader());
//        iTree.prepareForUse();
//        iTree.resetLearningImpl();
//        Evaluator avITree = new Evaluator(m, C_con, "ISOUPTree");
//        
//        //train only
//        for (int i = 0; i < train.size(); i++) {
//            InstanceExample inst = file.nextInstance();
//            EaHTps.trainOnInstance(inst);
//            EaBR.trainOnInstance(inst);
//            EaCC.trainOnInstance(inst);
//            PS.trainOnInstance(inst);
//            CC.trainOnInstance(inst);
//            MLHT.trainOnInstance(inst);
//            iTree.trainOnInstance(inst);
//        }
//
//        //Classificação
//        ArrayList<Prediction> predListEaHTps = new ArrayList<>();
//        ArrayList<Prediction> predListEaBR = new ArrayList<>();
//        ArrayList<Prediction> predListEaCC = new ArrayList<>();
//        ArrayList<Prediction> predListCC = new ArrayList<>();
//        ArrayList<Prediction> predListPS = new ArrayList<>();
//        ArrayList<Prediction> predListMLHT= new ArrayList<>();
//        ArrayList<Prediction> predListiTree= new ArrayList<>();
//        ArrayList<Set<String>> trueLabelsList = new ArrayList<>();
//        
//        for (int i = 1; i <= test.size(); i++) {
//            InstanceExample inst = file.nextInstance();
//            trueLabelsList.add(DataSetUtils.getLabelSet(inst.getData()));
//
//                //Test, evaluation and train EaHTps
//                predListEaHTps.add(EaHTps.getPredictionForInstance(inst));
//                EaHTps.trainOnInstance(inst);
//    
//                //Test, evaluation and train EaBR
//                predListEaBR.add(EaBR.getPredictionForInstance(inst));
//                EaBR.trainOnInstance(inst);
//
////                Test, evaluation and train EaCC
//                predListEaCC.add(EaCC.getPredictionForInstance(inst));
//                EaCC.trainOnInstance(inst);
//                
//               //Test, evaluation and train CC sem feedback
//                predListCC.add(CC.getPredictionForInstance(inst));
//
//    //            //Test, evaluation and train PS sem feedback
//                predListPS.add(PS.getPredictionForInstance(inst));
//                
//                predListMLHT.add(MLHT.getPredictionForInstance(inst));
//                
//                predListiTree.add(iTree.getPredictionForInstance(inst));
//
//                if (i % wEvaluation == 0){
//                    //If it have not label latency get the last window label cardinality
//                    avEaHTps.updateMeasures(predListEaHTps, windowsCardinalities[i/wEvaluation-1], trueLabelsList);
//                    avEaBR.updateMeasures(predListEaBR, windowsCardinalities[i/wEvaluation-1], trueLabelsList);
//                    avEaCC.updateMeasures(predListEaCC, windowsCardinalities[i/wEvaluation-1], trueLabelsList);
//
//                    //else get the train label cardinality
//                    avCC.updateMeasures(predListCC, cardinalityTrain, trueLabelsList);
//                    avPS.updateMeasures(predListPS, cardinalityTrain, trueLabelsList);
//                    avMLHT.updateMeasures(predListMLHT, cardinalityTrain, trueLabelsList);
//                    avITree.updateMeasures(predListiTree, cardinalityTrain, trueLabelsList);
//
//                    predListEaHTps.clear();
//                    predListEaBR.clear();
//                    predListEaCC.clear();
//                    predListCC.clear();
//                    predListPS.clear();
//                    predListMLHT.clear();
//                    predListiTree.clear();
//                    trueLabelsList.clear();
//                }
//                if(i == test.size() && (i % wEvaluation) > 0){
//                    avEaHTps.updateMeasures(predListEaHTps, windowsCardinalities[i/wEvaluation-1], trueLabelsList);
//                    avEaBR.updateMeasures(predListEaBR, windowsCardinalities[i/wEvaluation-1], trueLabelsList);
//                    avEaCC.updateMeasures(predListEaCC, windowsCardinalities[i/wEvaluation-1], trueLabelsList);
//
//                    //else get the train label cardinality
//                    avCC.updateMeasures(predListCC, cardinalityTrain, trueLabelsList);
//                    avPS.updateMeasures(predListPS, cardinalityTrain, trueLabelsList);
//                    avMLHT.updateMeasures(predListMLHT, cardinalityTrain, trueLabelsList);
//                    avITree.updateMeasures(predListiTree, cardinalityTrain, trueLabelsList);
//
//                    predListEaHTps.clear();
//                    predListEaBR.clear();
//                    predListEaCC.clear();
//                    predListCC.clear();
//                    predListPS.clear();
//                    predListMLHT.clear();
//                    predListiTree.clear();
//                    trueLabelsList.clear();
//                }
//        }
//        
//        av.add(avEaCC);
//        av.add(avEaHTps);
//        av.add(avEaBR);
//        av.add(avCC);
//        av.add(avPS);
//        av.add(avMLHT);
//        av.add(avITree);

        EvaluatorBR.writesAvgResults(av,outputDirectory);
        Evaluator.writeMeasuresOverTime(av, outputDirectory);
    }
    

    private static void removeClasses(String dataSetPath) throws IOException {
        MultiTargetArffFileStream stream = new MultiTargetArffFileStream(dataSetPath, "1");
        stream.prepareForUse();
        
        FileWriter dataSetFile = new FileWriter(new File(dataSetPath.replace(".arff", "-CE.arff")), false);
        
        //write the file's header
        dataSetFile.write("@relation \n");
        dataSetFile.write("@attribute att1 numeric \n");
        dataSetFile.write("@attribute att2 numeric \n");
        dataSetFile.write("@attribute class \n");
        dataSetFile.write("\n");
        dataSetFile.write("@data\n");
        
        int cont1 = 1;
        int cont2 = 1;
        while(cont1 <= 50000){
            Instance inst = stream.nextInstance().getData();
            System.out.println(inst.value(2) );
            if(inst.value(2) != 1.0 && inst.value(2) != 2.0){
                for (int i = 0; i < inst.numAttributes(); i++) {
                    dataSetFile.write(inst.value(i) + ",");
                }
                dataSetFile.write("\n");
            }
            cont1++;
            cont2++;
        }
        while(cont2 <= 100000){
            Instance inst = stream.nextInstance().getData();
            System.out.println(inst.value(2) );
            if(inst.value(2) != 2){
                for (int i = 0; i < inst.numAttributes(); i++) {
                    dataSetFile.write(inst.value(i) + ",");
                }
                dataSetFile.write("\n");
            }
            cont1++;
            cont2++;
        }
        while(stream.hasMoreInstances()){
            Instance inst = stream.nextInstance().getData();
            for (int i = 0; i < inst.numAttributes(); i++) {
                dataSetFile.write(inst.value(i) + ",");
            }
            dataSetFile.write("\n");
        }
        dataSetFile.close();
    }
    
}
