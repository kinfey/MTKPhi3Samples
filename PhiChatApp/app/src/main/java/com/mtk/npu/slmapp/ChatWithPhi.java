package com.mtk.npu.slmapp;


import android.app.Activity;
import android.content.res.AssetFileDescriptor;

import com.mediatek.neuropilot.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class ChatWithPhi {

    private static final String MODEL_PATH = "phi3.dla";

    protected Interpreter interpreter;
    ChatWithPhi(Activity activity) throws IOException {

        MappedByteBuffer file = loadModelFile(activity, MODEL_PATH);
        try {
        interpreter = new Interpreter(file);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

//        interpreter = new Interpreter(loadModelFile(activity, MODEL_PATH));
//        interpreter.setUseNNAPI(true);
    }
    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_PATH) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }




}
