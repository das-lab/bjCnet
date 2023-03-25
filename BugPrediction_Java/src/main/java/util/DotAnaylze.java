package util;

import com.kitfox.svg.A;
import guru.nidi.graphviz.model.Link;
import guru.nidi.graphviz.model.MutableGraph;
import guru.nidi.graphviz.model.MutableNode;
import guru.nidi.graphviz.model.Node;
import guru.nidi.graphviz.parse.Parser;

import org.apache.commons.lang3.StringEscapeUtils;
import util.StrProcess;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;


public class DotAnaylze {
    private String lineNum = "";
    private String rootNodeLabelStr = "";
    private ArrayList<HashMap> graphs = new ArrayList();

    public DotAnaylze(HashSet dotFilePaths) throws IOException {
        for (Object dotFilePath : dotFilePaths) {
            InputStream dot = new FileInputStream(dotFilePath.toString());
            MutableGraph g = new Parser().read(dot);

            Collection nodes = g.nodes();
            Collection edges = g.edges();

            ArrayList<HashMap> nodeInfos = this.parseNodes(nodes);
            ArrayList<ArrayList> edgeInfos = this.parseEdges(edges);

            if (this.lineNum.equals("")) {
                continue;
            }

            HashMap graphInfo = new HashMap();
            graphInfo.put("lineNum", this.lineNum);
            this.lineNum = "";
            graphInfo.put("rootNode",this.rootNodeLabelStr);
            this.rootNodeLabelStr = "";
            graphInfo.put("nodeInfos", nodeInfos);
            graphInfo.put("edgeInfos", edgeInfos);
            graphInfo.put("graphName", g.name().toString());

            this.graphs.add(graphInfo);
        }
    }

    public HashMap analyze(HashMap methodInfo) {
        for (HashMap graph : this.graphs) {
            if (graph.get("lineNum").equals(methodInfo.get("methodStartLine")) && graph.get("graphName").equals(methodInfo.get("methodName"))) {
                this.graphs.remove(graph);
                return graph;
            }
        }
        return null;
    }

    private void getMethodLineNumAndRootNode(String labelStr) {
        if (labelStr.indexOf("METHOD,") > -1 && labelStr.indexOf("<SUB>") > -1) {
            this.lineNum = labelStr.split("<SUB>")[1].split("</SUB>")[0];
            this.rootNodeLabelStr = labelStr.split("<SUB>")[0];
        }
    }

    private ArrayList<HashMap> parseNodes(Collection nodes) {
        ArrayList<HashMap> nodeInfos = new ArrayList<>();
        for (Object node : nodes) {
            MutableNode _node = (MutableNode) node;
            String nodeId = _node.name().toString();
            String labelStr = _node.attrs().get("label").toString();
            labelStr = StringEscapeUtils.unescapeHtml4(labelStr);

            if (this.lineNum.equals("")) {
                this.getMethodLineNumAndRootNode(labelStr);
            }

            labelStr = labelStr.split("<SUB>")[0];

            HashMap nodeInfo = new HashMap();
            nodeInfo.put("nodeId", nodeId);
            nodeInfo.put("labelStr", labelStr);

            nodeInfos.add(nodeInfo);
        }

        return nodeInfos;
    }

    private ArrayList<ArrayList> parseEdges(Collection edges) {
        ArrayList<ArrayList> edgeInfos = new ArrayList<>();
        for (Object e : edges) {
            Link _e = (Link) e;
            String src = _e.from().name().toString();
            String linkStr = _e.name().toString();
            String[] temp = linkStr.split("--");
            String dst = temp[1];

            ArrayList edgeInfo = new ArrayList();
            edgeInfo.add(src);
            edgeInfo.add(dst);

            edgeInfos.add(edgeInfo);
        }

        return edgeInfos;
    }
}
