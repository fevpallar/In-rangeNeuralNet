package org.fevly.neuralnet;
/*==================================
Author : Fevly P.
contact : fevly.pallar@gmail.com
====================================== */
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class CSVUtil {
    public static List<List<String>> records = new ArrayList<>();
    public  void fillRecords(String inputFileName) {
        try (BufferedReader br = new BufferedReader(new FileReader(inputFileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                records.add(Arrays.asList(values));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    public  int getTotalRows() {
        return records.size();
    }
    public void displayRecords (){
        System.out.println(records);
    }

}
