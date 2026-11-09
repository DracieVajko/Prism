import java.io.File
import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    // Keep serialization plugin for @Serializable annotations
    id("org.jetbrains.kotlin.plugin.serialization") version "1.9.24"
}

android {
    namespace = "eu.odysseus.diffusion"
    compileSdk = 34
    ndkVersion = "27.3.13750724"

    defaultConfig {
        applicationId = "eu.odysseus.diffusion"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    // Release signing is optional. When keystore.properties (or the matching
    // env vars) is absent the release variant is signed with the debug key so
    // `assembleRelease` still produces an installable APK for side-loading.
    // `java` resolves to the java extension here, so use the imported Properties.
    val keystoreProps = Properties().apply {
        val f = rootProject.file("keystore.properties")
        if (f.exists()) f.inputStream().use { load(it) }
    }
    val releaseStoreFile = (keystoreProps.getProperty("storeFile")
        ?: System.getenv("PRISM_KEYSTORE_FILE"))?.takeIf { it.isNotBlank() }?.let(::file)
    val hasReleaseKey = releaseStoreFile != null && releaseStoreFile.exists()

    signingConfigs {
        if (hasReleaseKey) {
            create("release") {
                storeFile = releaseStoreFile
                storePassword = keystoreProps.getProperty("storePassword")
                    ?: System.getenv("PRISM_KEYSTORE_PASSWORD")
                keyAlias = keystoreProps.getProperty("keyAlias")
                    ?: System.getenv("PRISM_KEY_ALIAS")
                keyPassword = keystoreProps.getProperty("keyPassword")
                    ?: System.getenv("PRISM_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            // R8 currently fails on the Ktor + Netty + ONNX Runtime + Gson
            // combination (com.android.build.gradle.internal.tasks.R8Task
            // "Compilation failed to complete") and there is no ProGuard
            // config covering those libraries yet. Shipping an installable,
            // un-minified APK is better than shipping a build that does not
            // exist; re-enable once proguard-rules.pro covers the stack and
            // the result has been smoke-tested on a device.
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = if (hasReleaseKey) {
                signingConfigs.getByName("release")
            } else {
                logger.warn("odysseus: no release keystore, signing release with the debug key")
                signingConfigs.getByName("debug")
            }
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        // litertlm-android 0.19.0 ships kotlin-stdlib/kotlin-reflect 2.4.0 whose
        // metadata is newer than this 1.9.24 compiler accepts by default; the
        // same flag is required by Google's own sample (litert-samples
        // receipt_to_json, which uses it with a 2.3.0 compiler).
        freeCompilerArgs = freeCompilerArgs + listOf("-Xskip-metadata-version-check")
    }
    packaging {
        resources {
            excludes += setOf("**/*.bin", "**/*.param", "**/*.gguf", "META-INF/INDEX.LIST", "META-INF/io.netty.versions.properties", "META-INF/DEPENDENCIES")
        }
    }

    sourceSets {
        getByName("main") {
            // Prebuilt ncnn shared lib (gitignored; see tools/fetch_ncnn.ps1).
            jniLibs.srcDir("src/main/cpp/third_party/ncnn-jni")
        }
    }

    val ndkPresent = findNdkDir() != null
    if (ndkPresent) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    } else {
        logger.warn("odysseus: NDK not found → Kotlin-only build (libwhisper.so will not be in the APK, audio uses the stub)")
    }
}

fun findNdkDir(): File? {
    val explicit = System.getenv("ANDROID_NDK_HOME")?.let(::file)
    if (explicit != null && explicit.isDirectory) return explicit
    val props = Properties()
    rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { props.load(it) }
    val sdkRoot = (props.getProperty("sdk.dir") as String?)?.let(::file)
        ?: System.getenv("ANDROID_HOME")?.let(::file)
        ?: System.getenv("ANDROID_SDK_ROOT")?.let(::file)
    val ndkRoot = sdkRoot?.resolve("ndk") ?: return null
    return ndkRoot.listFiles()?.firstOrNull { it.isDirectory }?.takeIf { it.resolve("build/cmake/android.toolchain.cmake").exists() }
        ?: ndkRoot.listFiles()?.firstOrNull { it.isDirectory }
}


dependencies {
    val ktor = "2.3.12"
    implementation("io.ktor:ktor-server-core:$ktor")
    implementation("io.ktor:ktor-server-netty:$ktor")
    implementation("io.ktor:ktor-server-content-negotiation:$ktor")
    implementation("io.ktor:ktor-server-status-pages:$ktor")
    // Use Gson for JSON parsing to avoid kotlinx.serialization version conflicts
    implementation("io.ktor:ktor-serialization-gson:$ktor")
    implementation("io.ktor:ktor-client-core:$ktor")
    implementation("io.ktor:ktor-client-okhttp:$ktor")
    implementation("com.google.code.gson:gson:2.10.1")
    // Keep kotlinx-serialization for @Serializable annotations (needed for data classes)
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.5.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.3")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    // SmolVLM native runtime (llama.cpp + libmtmd) in its own CMake project
    implementation(project(":smolvlm"))
    // OCR: Xiaomi-OCR-0 through LiteRT-LM (Kotlin API + liblitertlm_jni.so,
    // arm64-v8a, minSdk 24). Brings kotlin-stdlib/reflect 2.4.0 via its POM.
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.19.0")
    // EXIF orientation for OCR input photos (litertlm applies none itself).
    implementation("androidx.exifinterface:exifinterface:1.4.1")
    // Gemini Nano via ML Kit GenAI API (on-device summarization, cleanup, rewrite)
    // Optional: works on Pixel 8+, Samsung S24+, Xiaomi, Motorola with NPU support.
    // Falls back gracefully on unsupported devices (summarize returns 503).
    implementation("com.google.mlkit:genai-prompt:1.0.0-beta1")

    // ONNX Runtime GenAI: unified VLM runtime (replaces llama.cpp JNI)
    implementation("com.microsoft.onnxruntime:onnxruntime-genai:0.4.0")
}