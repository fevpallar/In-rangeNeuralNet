/*==================================
Author : Fevly P.
contact : fevly.pallar@gmail.com

NN scope : in range neural network (supplied points are within the range of training points)
NN context : approximation
====================================== */

package org.fevly.neuralnet;

import java.util.*;
import java.io.File;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.buffer.MemoryDataLoader;
import org.encog.ml.data.buffer.codec.CSVDataCODEC;
import org.encog.ml.data.buffer.codec.DataSetCODEC;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.csv.CSVFormat;

public class InRangeNeuralNetwork {
    // rentang normalisasi [-1,1]
    static double maxNorm = 1;
    static double minNorm = -1;


   /*========================================================
   Note #1

    Input vector 'training' & 'test' dan outputnya 'prediction'

    di normalisasi ke rentang [-1,1].
    Dengan cara mengambil nilai min dan max dari masing2 point ( training, test, prediksi).

    Alhasil, untuk mengembalikan vector ke skala awal (sebelum normalisasi)
    dibutuh nilai min dan max (dari masing2 input vector [train maupun test])
    untuk acuannya (rumusnya)
     ========================================================*/

    // Min-Max input/ouput vectors {test, training, prediksi}
    static double minRangeNorm = 0.00;
    static double maxRangeNorm = 5.00;

    static int numberOfPointsInTrainingVectors;

    static double normalizedInputPoint = 0.00, normalizedPredictionPoint = 0.00, normalizedTargetPoint = 0.00;


    static double valueDifference = 0.00;
    static int returnCode = 0;

    String trainFileName;
    static String networkFileName;

    static List<Double> denormalizedInputVector = new ArrayList<>(), denormalizedTargetVector = new ArrayList<>(), denormalizedPredictedVector = new ArrayList<>();


    private MLDataSet trainingSet;

            /*============================================
        See Note#1

        jadi training, test dan predicted vectors semuanya dinormalisasi
        dengan rentang min & max yang sama [0..5].

        Alhasil ini method bisa dipakai sekaligus untuk ketiga vector.

        Perihal haruskan direntang yg sama itu kurang tahu (bukan kurang tempe..)

        harusnya bisa..
        ============================================================================ */
    public static double denormalizeVector(double suppliedVect) {
        return ((minRangeNorm - maxRangeNorm) *
                suppliedVect - maxNorm * minRangeNorm +
                maxRangeNorm * minNorm) / (minNorm - maxNorm);
    }


    public InRangeNeuralNetwork(String trainFileName, int nInputNeuron, int nOutputNeuron) {

        CSVUtil csvUtil = new CSVUtil();
        csvUtil.fillRecords("D:/sampledata/normalized_train.csv");
        numberOfPointsInTrainingVectors = csvUtil.getTotalRows();


        this.trainFileName = trainFileName;
        this.trainingSet =
                loadInputVector(trainFileName, nInputNeuron, nOutputNeuron,
                        true, CSVFormat.ENGLISH, false);
    }

    public void saveNetworkOutput(String networkFileName) {
        this.networkFileName = networkFileName;
        File inputFile = new File(networkFileName);
        if (inputFile.exists())
            inputFile.delete();

        returnCode = 0;
        do {
            returnCode = trainValidateSaveNetwork();
        } while (returnCode > 0);
        Encog.getInstance().shutdown();

    }

    public static MLDataSet loadInputVector(String filename, int input, int ideal, boolean headers, CSVFormat format, boolean significance) {
        DataSetCODEC codec = new CSVDataCODEC(new File(filename), format, headers, input, ideal, significance);
        MemoryDataLoader load = new MemoryDataLoader(codec);
        MLDataSet dataset = load.external2Memory();
        return dataset;
    }



    public static void setupNetwork(BasicNetwork netModel, int nInputLayer, int nHiddenLayer, int nOutputLayer, int nNeuronsPerHiddenLayer) {

        for (int i = 0; i < nInputLayer; i++)
            netModel.addLayer(new BasicLayer(null, true, 1));
        for (int i = 0; i < nHiddenLayer; i++)
            netModel.addLayer(new BasicLayer(new ActivationTANH(), true, nNeuronsPerHiddenLayer));
        for (int i = 0; i < nOutputLayer; i++)
            netModel.addLayer(new BasicLayer(new ActivationTANH(), false, 1));
    }

    public static void epocProc(BasicNetwork netModel, MLDataSet trainingSet) {

        EncogDirectoryPersistence.saveObject(new File(networkFileName), netModel);


        System.out.println("Neural Network Output:");

        double minimumOfNormalizedDifference = 99999;
        double maximumOfNormalizedDifference = 0.00;

        double sumOfNormalizedDifference = 0.00;
        double averageOfNormalizedDifference = 0.00;

             for (MLDataPair pair : trainingSet) {
            // ini load ulang 'input' & 'expected value' sudah dikirm
            // ke NN sebelumnya via pair.getInput()
            // kenapa load ulang? Untuk keperluan denormalisasi
            // karena 'input & actual' data kan awal normalized

            MLData inputData = pair.getInput();// iterasi 1 :  [BasicMLData:-0.94]
            MLData actualData = pair.getIdeal();// iterasi 2 : [BasicMLData:-0.991]


            MLData predictData = netModel.compute(inputData);

            /*=======================================================
            denormValue= (minOriginal−maxOriginal)*normValue − Nh*minOriginal+maxOriginal* Nl / Nl-Nh
            =========================================================*/


            // MLData store data di basis index (dindex 0)
            normalizedInputPoint = inputData.getData(0);// 0= -0.94
            normalizedTargetPoint = actualData.getData(0); //0= -0.991
            normalizedPredictionPoint = predictData.getData(0); //0= -0.9908447408332587


            // denormalize actual/target data/prediction

            double denormInputXPointValue = denormalizeVector(normalizedInputPoint);
            double denormTargetXPointValue = denormalizeVector(normalizedTargetPoint);
            double denormPredictXPointValue = denormalizeVector(normalizedPredictionPoint);

            valueDifference = Math.abs(((denormTargetXPointValue -
                    denormPredictXPointValue) / denormTargetXPointValue) * 100.00);

            System.out.println("expected/target (original) = " + denormTargetXPointValue +
                    "  prediction (denormalized) = " + denormPredictXPointValue +
                    "  valueDifference = " + valueDifference);

            sumOfNormalizedDifference = sumOfNormalizedDifference + valueDifference;

            if (valueDifference < minimumOfNormalizedDifference)
                     minimumOfNormalizedDifference = valueDifference;
            if (valueDifference > maximumOfNormalizedDifference)
                maximumOfNormalizedDifference = valueDifference;


            denormalizedInputVector.add(denormInputXPointValue);
            denormalizedTargetVector.add(denormTargetXPointValue);
            denormalizedPredictedVector.add(denormPredictXPointValue);

        }

             try {
                 EncogDirectoryPersistence.saveObject(new File(networkFileName), netModel);
             } catch (Exception e){
                 System.out.println("Error saving network result");
             }


        averageOfNormalizedDifference = sumOfNormalizedDifference / (numberOfPointsInTrainingVectors -1);

        System.out.println(" ");
        System.out.println("min. of Error margin (%) = "+ minimumOfNormalizedDifference+ "\nmax. of Error margin (%) = " + maximumOfNormalizedDifference + "  \naverage of Error margin(%) = " + averageOfNormalizedDifference);
    }


    public int trainValidateSaveNetwork() {

        BasicNetwork network = new BasicNetwork();

        setupNetwork(network, 1, 9, 1, 5);
        network.getStructure().finalizeStructure();
        network.reset();

        // alternative lain pake Backpropagation
        final ResilientPropagation train = new ResilientPropagation(network, trainingSet);

        int epochCounter = 1;
        int returnCode;

        while (true) {
            train.iteration();
            System.out.println("Epoch #" + epochCounter + " Curr. Error:" + train.getError());

            epochCounter++;

            // restart lagi setelah 400 epoch error belum satisfy
            if (epochCounter >= 400 && network.calculateError(trainingSet) > 0.000000031) {
                returnCode = 1;
                System.out.println("Restart");
                return returnCode;
            }

            // exit sewaktu error < threshold
            if (network.calculateError(trainingSet) <= 0.00000002) {
                break;
            }
        }


        epocProc(network, trainingSet);

        returnCode = 0;
        return returnCode;
    }
}