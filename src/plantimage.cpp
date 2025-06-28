package com.example;

import java.awt.Component;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.MediaTracker;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InterruptedIOException;

public class PlantImage {
    public int[][][] data = new int[0][0][0];
    public String label = "";
    public int index = -1;
    public PlantImage(String fname, String plantName){ //fname can be relative or absolute
        if(fname.length()>2 && !fname.substring(0,3).equals("C:/")){
            fname = Main.datasetDirPath+fname; 
        }
        this.data = fileToImageArr(fname);
        this.label = plantName;
        String[] fnameSplit = fname.split("\\.|/");
        if(fnameSplit.length>1){
            this.index = Integer.parseInt(fnameSplit[fnameSplit.length-2]);
        }
        
    }

    final public int[][][] fileToImageArr(String fName){
        int[][][] result = new int[][][] {{{-1}}};
        try {
            long startTime = System.currentTimeMillis();
            //Supposedly faster than ImageIO
            DataInputStream dataInputStream = new DataInputStream(new FileInputStream(fName));
            byte inputBytes[] = new byte[dataInputStream.available()];
            dataInputStream.readFully(inputBytes);
            dataInputStream.close();
            Image image = Toolkit.getDefaultToolkit().createImage(inputBytes);
            MediaTracker tracker = new MediaTracker(new Component() {}); // dummy component
            tracker.addImage(image, 0);
            try {
                tracker.waitForID(0); //Toolkit is asynchronous and so wait for it
            } catch (InterruptedException e) {
                return new int[][][] {{{-1}}};
            }
            if (image.getWidth(null) < 0 || image.getHeight(null) < 0) {
                return new int[][][] {{{-1}}};
            }
            BufferedImage bImage = new BufferedImage(image.getWidth(null), image.getHeight(null), BufferedImage.TYPE_INT_RGB);
            //System.out.println("Getting image took "+String.valueOf(System.currentTimeMillis()-startTime)+"ms");
            Graphics2D g2d = bImage.createGraphics();
            g2d.drawImage(image, 0, 0, null);  // Copies the pixels onto bImage
            g2d.dispose();
            int width = bImage.getWidth();
            int height = bImage.getHeight();
            
            result = new int[3][height][width]; //RGB
            //There is no alpha as most images in this dataset are jpeg which don't have an alpha channel
            int pixel;
            long parsingStart = System.currentTimeMillis();
            for (int y=0;y<height;y++) {
                for (int x=0;x<width;x++) {
                    pixel = bImage.getRGB(x,y);
                    int r =(pixel & 0xff0000) >> 16;
                    int g = (pixel & 0xff00) >> 8;
                    int b = pixel & 0xff;
                    result[0][y][x] = r; //R
                    result[1][y][x] = g;//G
                    result[2][y][x] = b; //B
                    //Using AND first stops sign issues 
                }
            }
            //System.out.println("Parsing image took "+String.valueOf(System.currentTimeMillis()-parsingStart)+"ms");
            return result;
        } 
        catch(InterruptedIOException e){
            System.out.println("interrupted");
            return new int[][][] {{{-1}}};    
        }
        catch (IOException ex) {
            //System.err.println(ex.toString()); If we have the wrong file extension
            return new int[][][] {{{-1}}};        
        }
    }
}
