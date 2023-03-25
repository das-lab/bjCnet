package scripts;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;

import util.Config;

public class RunJoern {
    public static void main(String[] args) {
        // use joern-cli by command to generate CPG
        String parseBatStr = "XXX\\joern-parse.bat %s"; // XXX is joern-parse.bat's dir
        String exportBatStr = "XXX\\joern-export.bat --repr=ast --format=dot --out=%s"; // XXX is joern-parse.bat's dir
        String sourceCodePath = Config.processedDataPath + File.separator + Config.product;
        String dotASTPatch = Config.dotASTPath;

        File file = new File(sourceCodePath);
        File[] files = file.listFiles();

        for (File f : files) {
            System.out.println("---------processing：" + f.getAbsolutePath() + " ---------");
            sourceCodePath = f.getAbsolutePath();
            String bugFilePath = sourceCodePath + File.separator + "bug";
            String pureCleanFilePath = sourceCodePath + File.separator + "pure_clean";

            RunJoern.process(bugFilePath,pureCleanFilePath, parseBatStr, exportBatStr, dotASTPatch);
        }
    }

    private static void process(String bugFilePath, String pureCleanFilePath, String parseBatStr, String exportBatStr, String dotASTPatch){
        HashSet sourceCodeFiles = new HashSet();

        if(!bugFilePath.equals("")){
            RunJoern.folderTraversal(new File(bugFilePath), sourceCodeFiles);
        }
        if(!pureCleanFilePath.equals("")){
            RunJoern.folderTraversal(new File(pureCleanFilePath), sourceCodeFiles);

        }

        int completed = 0;
        for (Object filePath : sourceCodeFiles) {
            String _filePath = filePath.toString();
            // parse
            System.out.println(String.format("parse：%s", _filePath));
            String cmdStr = String.format(parseBatStr, _filePath);
            RunJoern.runCmd(cmdStr);

            // export
            System.out.println("export AST");
            String[] temp = _filePath.split("\\\\");
            String tempGraphPath = dotASTPatch;
            for (int i = 7; i < temp.length - 1; i++) {
                tempGraphPath += File.separator + temp[i];
            }
            File tempFile = new File(tempGraphPath);
            if (!tempFile.exists()) {
                tempFile.mkdirs();
            }
            cmdStr = String.format(exportBatStr, tempGraphPath + File.separator + temp[temp.length - 1]);
            RunJoern.runCmd(cmdStr);

            completed += 1;
            System.out.println("remain：" + (sourceCodeFiles.size() - completed));
        }
    }

    private static void runCmd(String cmdStr) {
        BufferedReader br = null;
        try {
            File file = new File(Config.tempFilePath);
            File tmpFile = new File(Config.tempFilePath + File.separator + "temp.tmp");
            if (!file.exists()) {
                file.mkdirs();
            }
            if (!tmpFile.exists()) {
                tmpFile.createNewFile();
            }
            ProcessBuilder pb = new ProcessBuilder().command("cmd.exe", "/c", cmdStr).inheritIO();
            pb.redirectErrorStream(true);
            pb.redirectOutput(tmpFile);
            pb.start().waitFor();
            InputStream in = new FileInputStream(tmpFile);
            br = new BufferedReader(new InputStreamReader(in));
            String line = null;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
            br.close();
            br = null;
            tmpFile.delete();
            System.out.println("cmd over");
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private static void folderTraversal(File file, HashSet sourceCodeFiles) {
        File[] fs = file.listFiles();
        for (File f : fs) {
            if (f.isDirectory()) {
                RunJoern.folderTraversal(f, sourceCodeFiles);
            }
            if (f.isFile()) {
                String path = f.getAbsolutePath().replace("\\bug.java", "");
                path = path.replace("\\patch.java", "");
                path = path.replace("\\pureclean.java","");

                sourceCodeFiles.add(path);
            }
        }
    }
}
