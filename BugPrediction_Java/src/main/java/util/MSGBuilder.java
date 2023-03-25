package util;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.NumberFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.github.difflib.*;
import com.github.difflib.patch.Patch;

import com.kitfox.svg.A;
import util.DotAnaylze;

public class MSGBuilder {
    // Build Method Scope Graph
    private static HashMap nodeFeatures = new HashMap();
    private static ArrayList edgeIndicators = new ArrayList();
    // edge type：(1) package [1,0,0,0,0,0,0], (2) import [0,1,0,0,0,0,0], (3) field, (4) method
    // (5) implements&extends, (6) ast, (7) ddg
    private static HashMap edgeFeatures = new HashMap();
    // 单独存储csg图
    private static HashMap nodeFeatures4csg;
    private static ArrayList edgeIndicators4csg;
    private static HashMap edgeFeatures4csg;

    public static HashMap fFeat2fName = new HashMap();
    public static HashMap mFeat2mName = new HashMap();
    private static String label = "-1";

    private static HashMap graph2method = new HashMap();

    private static HashSet isVisited = new HashSet();

    private static ArrayList getSourceCodePath(String processedDataPath, String product) {
        ArrayList sourceCodeFiles = new ArrayList();

        String targetPath = processedDataPath + File.separator + product;
        File file = new File(targetPath);
        MSGBuilder.folderTraversal(file, sourceCodeFiles);

        return sourceCodeFiles;
    }

    private static void folderTraversal(File file, ArrayList files) {
        File[] fs = file.listFiles();
        for (File f : fs) {
            if (f.isDirectory()) {
                MSGBuilder.folderTraversal(f, files);
            }
            if (f.isFile()) {
                files.add(f.getAbsolutePath());
            }
        }
    }

    private static void getVersion(String sourceCodePath) {
        String[] temp = sourceCodePath.split("\\\\");
        Config.version = temp[8];
    }

    private static String getRelativePathAndLabel(String sourceCodePath) {
        String relativePath = Config.product;
        String[] temp = sourceCodePath.split(Config.product);
        String[] temp1 = temp[1].split("\\\\");

        for (int i = 1; i < temp1.length - 1; i++) {
            relativePath += File.separator + temp1[i];
        }

        if (temp1[temp1.length - 1].indexOf("bug") > -1) {
            MSGBuilder.label = "1";
        } else {
            MSGBuilder.label = "0";
        }

        return relativePath;
    }

    private static void parseCode(String sourceCodePath) throws IOException {
        VoidVisitor<List<HashMap>> methodVisitor = new MethodVisitor();
        VoidVisitor<List<HashMap>> classVisitor = new ClassVisitor();

        if (MSGBuilder.isVisited.contains(sourceCodePath)) {
            return;
        } else {
            MSGBuilder.isVisited.add(sourceCodePath);
        }

        CompilationUnit cu = null;
        try {
            cu = StaticJavaParser.parse(new File(sourceCodePath));
        } catch (Exception e) {
            MSGBuilder.clearGlobalField();
            return;
        }

        List<HashMap> classList = new ArrayList<>();
        classVisitor.visit(cu, classList);
        List<HashMap> methodInfoList_src = new ArrayList<>();
        methodVisitor.visit(cu, methodInfoList_src);

        String relativePath = MSGBuilder.getRelativePathAndLabel(sourceCodePath);
        MSGBuilder.CSGBuild(cu, classList, relativePath, methodInfoList_src);

        if (sourceCodePath.indexOf("bug.") > -1 || sourceCodePath.indexOf("patch.") > -1) {
            String corCodePath = MSGBuilder.getCorCodePath(sourceCodePath);
            if (MSGBuilder.isVisited.contains(corCodePath)) {
                return;
            } else {
                MSGBuilder.isVisited.add(corCodePath);
            }

            MSGBuilder.getRelativePathAndLabel(corCodePath);
            CompilationUnit cu_cor = null;
            try {
                cu_cor = StaticJavaParser.parse(new File(corCodePath));
            } catch (Exception e) {
                MSGBuilder.clearGlobalField();
                return;
            }


            classList = new ArrayList<>();
            classVisitor.visit(cu_cor, classList);
            List<HashMap> methodInfoList_patch = new ArrayList<>();
            methodVisitor.visit(cu_cor, methodInfoList_patch);

            relativePath = MSGBuilder.getRelativePathAndLabel(corCodePath);
            MSGBuilder.CSGBuild(cu_cor, classList, relativePath, methodInfoList_patch);

            if (methodInfoList_patch.size() > 0) {
                int bugFirst = 0;
                if (sourceCodePath.indexOf("bug\\bug.") > -1) {
                    bugFirst = 1;
                }

                ArrayList methodPairList = MSGBuilder.pairMethod(methodInfoList_src, methodInfoList_patch, bugFirst);
                if (methodPairList.size() == 0) {
                    return;
                }
                MSGBuilder.getGraphMapping(methodPairList);
            } else {
                return;
            }

            MSGBuilder.graph2method.clear();
        }
    }

    private static void clearGlobalField() {
        MSGBuilder.nodeFeatures.clear();
        MSGBuilder.edgeIndicators.clear();
        MSGBuilder.edgeFeatures.clear();
        MSGBuilder.nodeFeatures4csg = null;
        MSGBuilder.edgeFeatures4csg = null;
        MSGBuilder.edgeIndicators4csg = null;
        MSGBuilder.fFeat2fName.clear();
        MSGBuilder.mFeat2mName.clear();
        MSGBuilder.graph2method.clear();
    }

    private static void getGraphMapping(ArrayList methodPairList) {
        ArrayList graphPairList = new ArrayList();

        ArrayList bugKey = new ArrayList();
        ArrayList patchKey = new ArrayList();
        for (Object key : MSGBuilder.graph2method.keySet()) {
            if (key.toString().indexOf("bug") > -1) {
                bugKey.add(key);
            } else {
                patchKey.add(key);
            }
        }

        for (Object methodPair : methodPairList) {
            HashMap bugMethod = (HashMap) ((ArrayList) methodPair).get(0);
            HashMap patchMethod = (HashMap) ((ArrayList) methodPair).get(1);

            for (Object key : bugKey) {
                String graphPathBug = MSGBuilder.travesalGM(bugMethod, key.toString());
                if (graphPathBug != null) {
                    for (Object key1 : patchKey) {
                        String graphPathPatch = MSGBuilder.travesalGM(patchMethod, key1.toString());
                        if (graphPathPatch != null) {
                            graphPairList.add(graphPathBug + "|" + graphPathPatch);
                        }
                    }
                }
            }
        }

        String content = "";
        for (Object graphPair : graphPairList) {
            content += graphPair.toString() + "\n";
        }

        MSGBuilder.writeFile(content, Config.graphMapFile);
    }

    private static String travesalGM(HashMap method, String key) {
        String[] values = key.split("_");
        HashMap info = (HashMap) MSGBuilder.graph2method.get(key);
        String methodDeclFromG2M = info.get("methodDecl").toString();

        String methodStartLineFromMethodPair;
        String methodNameFromMethodPair;
        String methodDeclFromMethodPair;
        methodStartLineFromMethodPair = method.get("methodStartLine").toString();
        methodNameFromMethodPair = method.get("methodName").toString();
        methodDeclFromMethodPair = method.get("methodDecl").toString();

        if (values[0].equals(methodNameFromMethodPair) && values[1].equals(methodStartLineFromMethodPair) && methodDeclFromG2M.equals(methodDeclFromMethodPair)) {
            return info.get("graphPath").toString();
        }

        return null;
    }

    private static ArrayList CSGBuild(CompilationUnit cu, List<HashMap> classList, String relativePath, List<HashMap> methodInfoList_src) throws IOException {
        ArrayList csgList = new ArrayList();
        for (HashMap c : classList) {
            String className = c.get("className").toString();

            int nodeIndex = MSGBuilder.nodeFeatures.size() + 1;
            MSGBuilder.nodeFeatures.put(String.valueOf(nodeIndex), className.replace("\n", " ").replace("\r\n", " ").replace("\r", " "));

            String packageDecl = cu.getPackageDeclaration().get().toString();
            MSGBuilder.setNodeAndEdge(packageDecl.replace("\r\n", "").replace(";", ""), "1,0,0,0,0,0,0", "1");

            NodeList importDecl = cu.getImports();
            for (int i = 0; i < importDecl.size(); i++) {
                MSGBuilder.setNodeAndEdge(cu.getImport(i).toString().replace("\r\n", "").replace(";", ""), "0,1,0,0,0,0,0", "1");
            }

            ArrayList fieldList = (ArrayList) c.get("fieldList");
            for (Object f : fieldList) {
                MSGBuilder.setNodeAndEdge(f.toString(), "0,0,1,0,0,0,0", "1");
            }

            ArrayList ieList = (ArrayList) c.get("ieList");
            for (Object ie : ieList) {
                MSGBuilder.setNodeAndEdge(ie.toString(), "0,0,0,0,1,0,0", "1");
            }

            MSGBuilder.nodeFeatures4csg = (HashMap) MSGBuilder.nodeFeatures.clone();
            MSGBuilder.edgeIndicators4csg = (ArrayList) MSGBuilder.edgeIndicators.clone();
            MSGBuilder.edgeFeatures4csg = (HashMap) MSGBuilder.edgeFeatures.clone();

            MSGBuilder.MSGBuild(methodInfoList_src, relativePath, className, (ArrayList) c.get("methodList"));

            MSGBuilder.nodeFeatures.clear();
            MSGBuilder.edgeIndicators.clear();
            MSGBuilder.edgeFeatures.clear();
        }

        return csgList;
    }

    private static void MSGBuild(List<HashMap> methodInfoList, String relativePath, String className, ArrayList methodList) throws IOException {
        String dotASTPath = Config.dotASTPath + File.separator + Config.product;
        ArrayList dotFiles = new ArrayList();
        MSGBuilder.folderTraversal(new File(dotASTPath), dotFiles);

        HashSet _dotFiles = new HashSet();
        for (Object dotFile : dotFiles) {
            String _dotFile = dotFile.toString();
            if (_dotFile.indexOf(relativePath) > -1) {
                _dotFiles.add(_dotFile);
            }
        }
        DotAnaylze dotAnaylze = new DotAnaylze(_dotFiles);

        for (HashMap methodInfo : methodInfoList) {
            if (className.equals(methodInfo.get("className"))) {
                HashMap ast = dotAnaylze.analyze(methodInfo);
                if (ast != null) {
                    for (Object m : methodList) {
                        if (!m.toString().equals(methodInfo.get("methodDecl").toString())) {
                            MSGBuilder.setNodeAndEdge(m.toString(), "0,0,0,1,0,0,0", "1");
                        }
                    }

                    MSGBuilder.setNodeAndEdge(ast.get("rootNode").toString(), "0,0,0,0,0,0,1", "1");

                    ArrayList nodeInfos = (ArrayList) ast.get("nodeInfos");
                    ArrayList edgeInfos = (ArrayList) ast.get("edgeInfos");
                    for (Object nodeInfo : nodeInfos) {
                        HashMap _nodeInfo = (HashMap) nodeInfo;

                        String srcId = MSGBuilder.dataDependenceAna(_nodeInfo.get("labelStr").toString());
                        if (srcId != null) {
                            MSGBuilder.setNodeAndEdge(_nodeInfo.get("labelStr").toString(), "0,0,0,0,0,0,1", srcId);
                        }

                        ArrayList relatedEdges = MSGBuilder.getRelatedEdges(edgeInfos, nodeInfos, _nodeInfo);
                        for (Object relatedEdge : relatedEdges) {
                            ArrayList _relatedEdge = (ArrayList) relatedEdge;
                            String src = _relatedEdge.get(0).toString();
                            String dst = _relatedEdge.get(1).toString();

                            srcId = MSGBuilder.getSrcId(src);
                            if (srcId.equals("")) {
                                srcId = String.valueOf(MSGBuilder.nodeFeatures.size() + 1);
                                MSGBuilder.nodeFeatures.put(srcId, src.replace("\n", " ").replace("\r\n", " ").replace("\r", " "));
                            }

                            MSGBuilder.setNodeAndEdge(dst, "0,0,0,0,0,1,0", srcId);
                        }
                    }

                    String graphName = ast.get("graphName") + "_" + ast.get("lineNum");
                    String methodDecl = methodInfo.get("methodDecl").toString();
                    MSGBuilder.saveMSG(relativePath, graphName, methodDecl);

                    MSGBuilder.nodeFeatures = (HashMap) MSGBuilder.nodeFeatures4csg.clone();
                    MSGBuilder.edgeIndicators = (ArrayList) MSGBuilder.edgeIndicators4csg.clone();
                    MSGBuilder.edgeFeatures = (HashMap) MSGBuilder.edgeFeatures4csg.clone();
                }
            }
        }
    }

    private static void saveMSG(String relativePath, String graphName, String methodDecl) throws IOException {
        String graphPath = Config.msgPath;
        String[] temp = relativePath.split("\\\\");
        int len4for = temp.length - 1;
        if (relativePath.indexOf("pure_clean") > -1) {
            len4for = temp.length;
        }
        for (int i = 0; i < len4for; i++) {
            graphPath += File.separator + temp[i];
        }

        graphPath += File.separator + graphName;
        if (MSGBuilder.label.equals("1")) {
            graphPath += "_bug";
        } else {
            if (relativePath.indexOf("pure_clean") < 0) {
                graphPath += "_patch";
            }
        }

        temp = graphPath.split("\\\\");
        HashMap info = new HashMap();
        info.put("graphPath", graphPath);
        info.put("methodDecl", methodDecl);
        MSGBuilder.graph2method.put(temp[temp.length - 1], info);

        File file = new File(graphPath);
        if (!file.exists()) {
            file.mkdirs();
        }

        String adjacentMatrixPath = graphPath + File.separator + "AM.txt";
        file = new File(adjacentMatrixPath);
        if (!file.exists()) {
            file.createNewFile();
        }
        String content = "";
        for (Object edge : MSGBuilder.edgeIndicators) {
            ArrayList _edge = (ArrayList) edge;
            content += _edge.get(0).toString() + "," + _edge.get(1).toString() + "\n";
        }
        content = content.substring(0, content.length() - 1);
        MSGBuilder.writeFile(content, adjacentMatrixPath);

        String edgeFeaturesPath = graphPath + File.separator + "edge_features.txt";
        file = new File(edgeFeaturesPath);
        if (!file.exists()) {
            file.createNewFile();
        }
        content = "";
        for (int i = 0; i < MSGBuilder.edgeFeatures.size(); i++) {
            Object edgeFeature = MSGBuilder.edgeFeatures.get(String.valueOf(i + 1));
            if (edgeFeature == null) {
                System.out.println("edgeId 错误");
            }
            content += edgeFeature.toString() + "\n";
        }
        content = content.substring(0, content.length() - 1);
        MSGBuilder.writeFile(content, edgeFeaturesPath);


        String nodeFeaturesPath = graphPath + File.separator + "node_features_raw.txt";
        file = new File(nodeFeaturesPath);
        if (!file.exists()) {
            file.createNewFile();
        }
        content = "";
        for (int i = 0; i < MSGBuilder.nodeFeatures.size(); i++) {
            Object nodeFeature = MSGBuilder.nodeFeatures.get(String.valueOf(i + 1));
            if (nodeFeature == null) {
                System.out.println("nodeId 错误");
            }
            content += nodeFeature.toString() + "\n";
        }
        content = content.substring(0, content.length() - 1);
        MSGBuilder.writeFile(content, nodeFeaturesPath);

        MSGBuilder.nodeFeatures.clear();
        MSGBuilder.edgeFeatures.clear();
        MSGBuilder.edgeIndicators.clear();
    }

    private static void writeFile(String content, String filePath) {
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(filePath, true))) {
            bufferedWriter.write(content);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String dataDependenceAna(String labelStr) {
        for (Object key : MSGBuilder.fFeat2fName.keySet()) {
            String value = MSGBuilder.fFeat2fName.get(key).toString();

            String patternStr = String.format("(?<![a-zA-Z0-9_])%s", value);
            Pattern pattern = Pattern.compile(patternStr);
            Matcher matcher = pattern.matcher(labelStr);
            if (matcher.find()) {
                for (Object key1 : MSGBuilder.nodeFeatures.keySet()) {
                    String nodeFeature = MSGBuilder.nodeFeatures.get(key1).toString();
                    if (nodeFeature.equals(key.toString())) {
                        return key1.toString();
                    }
                }
            }
        }

        return null;
    }

    private static String getSrcId(String nodeFeature) {
        String srcId = "";
        if (!MSGBuilder.nodeFeatures.values().contains(nodeFeature)) {
            return "";
        }

        for (Object key : MSGBuilder.nodeFeatures.keySet()) {
            String value = MSGBuilder.nodeFeatures.get(key).toString();
            if (value.equals(nodeFeature)) {
                return key.toString();
            }
        }

        return srcId;
    }

    private static void setNodeAndEdge(String tgtNodeFeature, String edgeFeature, String srcNodeId) {
        String nodeIndex = "";
        if (MSGBuilder.nodeFeatures.values().contains(tgtNodeFeature)) {
            for (Object key : MSGBuilder.nodeFeatures.keySet()) {
                String value = MSGBuilder.nodeFeatures.get(key).toString();
                if (value.equals(tgtNodeFeature)) {
                    nodeIndex = key.toString();
                }
            }
        } else {
            nodeIndex = String.valueOf(MSGBuilder.nodeFeatures.size() + 1);

            MSGBuilder.nodeFeatures.put(nodeIndex, tgtNodeFeature.replace("\n", " ").replace("\r\n", " ").replace("\r", " "));
        }

        for (int i = 0; i < MSGBuilder.edgeIndicators.size(); i++) {
            ArrayList edge = (ArrayList) MSGBuilder.edgeIndicators.get(i);
            String edgeFeatures = MSGBuilder.edgeFeatures.get(String.valueOf(i + 1)).toString();
            if (srcNodeId.equals(edge.get(0)) && nodeIndex.equals(edge.get(1)) && edgeFeatures.equals(edgeFeature)) {
                return;
            }
        }

        int edgeIndex = MSGBuilder.edgeIndicators.size() + 1;
        ArrayList edge = new ArrayList();
        edge.add(srcNodeId);
        edge.add(nodeIndex);
        MSGBuilder.edgeIndicators.add(edge);
        MSGBuilder.edgeFeatures.put(String.valueOf(edgeIndex), edgeFeature);
    }

    private static ArrayList getRelatedEdges(ArrayList edgeInfos, ArrayList nodeInfos, HashMap node) {
        ArrayList relatedEdges = new ArrayList();
        for (Object edge : edgeInfos) {
            ArrayList _edge = (ArrayList) edge;
            String relatedNodeLabelStr = "";
            if (_edge.get(0).toString().equals(node.get("nodeId"))) {
                String tgtNodeId = _edge.get(1).toString();
                relatedNodeLabelStr = MSGBuilder.getRelatedNode(nodeInfos, tgtNodeId);

                if (relatedNodeLabelStr != null && !relatedNodeLabelStr.equals("")) {
                    ArrayList relatedEdge = new ArrayList();
                    relatedEdge.add(node.get("labelStr"));
                    relatedEdge.add(relatedNodeLabelStr);
                    relatedEdges.add(relatedEdge);
                }
            }
            if (_edge.get(1).toString().equals(node.get("nodeId"))) {
                String srcNodeId = _edge.get(0).toString();
                relatedNodeLabelStr = MSGBuilder.getRelatedNode(nodeInfos, srcNodeId);

                if (relatedNodeLabelStr != null && !relatedNodeLabelStr.equals("")) {
                    ArrayList relatedEdge = new ArrayList();
                    relatedEdge.add(relatedNodeLabelStr);
                    relatedEdge.add(node.get("labelStr"));
                    relatedEdges.add(relatedEdge);
                }
            }
        }

        return relatedEdges;
    }

    private static String getRelatedNode(ArrayList nodeInfos, String nodeId) {
        for (Object node : nodeInfos) {
            HashMap _node = (HashMap) node;
            if (_node.get("nodeId").toString().equals(nodeId)) {
                return _node.get("labelStr").toString();
            }
        }

        return null;
    }

    private static Boolean isDiff(ArrayList methodPair) {
        String bugMethod = ((HashMap) methodPair.get(0)).get("methodDecl") + "\n" + ((HashMap) methodPair.get(0)).get("methodBody");
        String patchMethod = ((HashMap) methodPair.get(1)).get("methodDecl") + "\n" + ((HashMap) methodPair.get(1)).get("methodBody");

        List<String> bugMethodLines = Arrays.asList(bugMethod.split("\n"));
        List<String> patchMethodLines = Arrays.asList(patchMethod.split("\n"));

        Patch<String> patch = DiffUtils.diff(bugMethodLines, patchMethodLines);

        if (patch.getDeltas().size() > 0) {
            return true;
        }

        return false;
    }

    private static ArrayList pairMethod(List<HashMap> methodInfoList_src, List<HashMap> methodInfoList_patch, int bugFirst) {
        ArrayList methodPairList = new ArrayList();

        ArrayList pureCleanList = new ArrayList();
        ArrayList B2PList = new ArrayList();
        for (Object methodInfo_src : methodInfoList_src) {
            HashMap _methodInfo_src = (HashMap) methodInfo_src;
            String methodDecl_src = _methodInfo_src.get("methodDecl").toString();
            int tempIdx = methodDecl_src.lastIndexOf(")");
            methodDecl_src = methodDecl_src.substring(0, tempIdx + 1);

            for (Object methodInfo_patch : methodInfoList_patch) {
                HashMap _methodInfo_patch = (HashMap) methodInfo_patch;
                String methodDecl_patch = _methodInfo_patch.get("methodDecl").toString();
                tempIdx = methodDecl_patch.lastIndexOf(")");
                methodDecl_patch = methodDecl_patch.substring(0, tempIdx + 1);

                if (methodDecl_src.equals(methodDecl_patch)) {
                    ArrayList methodPair = new ArrayList();

                    if (bugFirst == 1) {
                        methodPair.add(methodInfo_src);
                        methodPair.add(methodInfo_patch);
                    } else {
                        methodPair.add(methodInfo_patch);
                        methodPair.add(methodInfo_src);
                    }

                    if (MSGBuilder.isDiff(methodPair)) {
                        B2PList.add(methodPair);
                    } else {
                        pureCleanList.add(methodInfo_src);
                    }

                    break;
                }
            }
        }

        for (int i = 0; i < B2PList.size(); i++) {
            Random random = new Random();
            ArrayList methodPair = new ArrayList();

            ArrayList temp = (ArrayList) B2PList.get(i);
            methodPair.add(temp.get(0));
            methodPair.add(temp.get(1));
            if (pureCleanList.size() == 0) {
                methodPair.add(null);
            } else {
                methodPair.add(pureCleanList.get(random.nextInt(pureCleanList.size())));
            }
            methodPairList.add(methodPair);
        }

        return methodPairList;
    }

    private static String getCorCodePath(String sourceCodePath) {
        String corCodePath = "";
        if (sourceCodePath.indexOf("bug.") > -1) {
            corCodePath = sourceCodePath.replace("bug\\bug.", "patch\\patch.");
        } else {
            corCodePath = sourceCodePath.replace("patch\\patch.", "bug\\bug.");
        }

        return corCodePath;
    }

    private static class MethodVisitor extends VoidVisitorAdapter<List<HashMap>> {
        @Override
        public void visit(MethodDeclaration md, List<HashMap> collector) {
            super.visit(md, collector);

            if (md.getParentNode().get() instanceof ClassOrInterfaceDeclaration) {
                int methodStartLine = md.getRange().get().begin.line;

                String methodDecl = md.getDeclarationAsString();
                if (md.getBody().isEmpty()) {
                    return;
                }
                String methodBody = md.getBody().get().toString();
                String methodName = md.getNameAsString();

                HashMap methodInfo = new HashMap();
                methodInfo.put("methodStartLine", String.valueOf(methodStartLine));
                methodInfo.put("methodDecl", methodDecl);
                methodInfo.put("methodBody", methodBody);
                methodInfo.put("methodName", methodName);

                ClassOrInterfaceDeclaration classInfo = (ClassOrInterfaceDeclaration) md.getParentNode().get();
                methodInfo.put("className", classInfo.getFullyQualifiedName().get());

                collector.add(methodInfo);
            }
        }
    }

    private static class ClassVisitor extends VoidVisitorAdapter<List<HashMap>> {
        @Override
        public void visit(ClassOrInterfaceDeclaration cid, List<HashMap> collector) {
            super.visit(cid, collector);
            HashMap classInfo = new HashMap();
            classInfo.put("className", cid.getFullyQualifiedName().get());

            ArrayList ieList = new ArrayList();
            for (Object i : cid.getImplementedTypes()) {
                ClassOrInterfaceType _i = (ClassOrInterfaceType) i;
                ieList.add(_i.getNameWithScope().replace(";", ""));
            }
            for (Object i : cid.getExtendedTypes()) {
                ClassOrInterfaceType _i = (ClassOrInterfaceType) i;
                ieList.add(_i.getNameWithScope().replace(";", ""));
            }

            ArrayList methodList = new ArrayList();
            for (Object m : cid.getMethods().toArray()) {
                MethodDeclaration _m = (MethodDeclaration) m;
                methodList.add(_m.getDeclarationAsString().replace(";", ""));

                MSGBuilder.mFeat2mName.put(_m.getDeclarationAsString().replace(";", ""), _m.getNameAsString());
            }
            for (Object m : cid.getConstructors().toArray()) {
                ConstructorDeclaration _m = (ConstructorDeclaration) m;
                methodList.add(_m.getNameAsString().replace(";", ""));

                MSGBuilder.mFeat2mName.put(_m.getDeclarationAsString().replace(";", ""), _m.getNameAsString());
            }

            ArrayList fieldList = new ArrayList();
            for (Object v : cid.getFields().toArray()) {
                FieldDeclaration _v = (FieldDeclaration) v;
                fieldList.add(_v.toString().replace(";", ""));

                MSGBuilder.fFeat2fName.put(_v.toString().replace(";", ""), _v.getVariables().get(0).getName().toString());
            }

            classInfo.put("methodList", methodList);
            classInfo.put("fieldList", fieldList);
            classInfo.put("ieList", ieList);

            collector.add(classInfo);
        }
    }

    public static void main(String[] agrs) throws IOException, InterruptedException {
        String product = Config.product;
        String processedDataPath = Config.processedDataPath;

        ArrayList sourceCodeFiles = MSGBuilder.getSourceCodePath(processedDataPath, product);
        LinkedHashSet linkedHashSet = new LinkedHashSet(sourceCodeFiles);
        sourceCodeFiles = new ArrayList(linkedHashSet);

        int completed = 0;
        for (Object path : sourceCodeFiles) {
            try{
                MSGBuilder.parseCode(path.toString());
            }
            catch (Exception e){
                System.out.println(String.format("parse error\n%s",path.toString()));
            }
            completed++;
            System.out.println(String.format("remain %s files need to process", String.valueOf(sourceCodeFiles.size() - completed)));
        }
    }
}
