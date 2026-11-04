package eu.odysseus.diffusion

import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.TextView
import java.io.File

/**
 * Adapter for the Outputs dialog list.
 * Shows items grouped by type with section headers.
 * Each item shows: checkbox, thumbnail (for images), name, size, delete button.
 */
class OutputsAdapter(
    private val context: Context,
    private val items: MutableList<OutputStore.OutputFile>,
    private val onItemClick: (OutputStore.OutputFile) -> Unit,
    private val onDelete: (OutputStore.OutputFile) -> Unit,
) : ArrayAdapter<OutputStore.OutputFile>(context, 0, items) {

    private val inflater: LayoutInflater = LayoutInflater.from(context)

    // Section headers: maps position -> section header text
    private val sectionPositions: MutableMap<Int, String> = mutableMapOf()

    init {
        rebuildSections()
    }

    /** Rebuild section headers based on current items list. */
    fun rebuildSections() {
        sectionHeaders.clear()
        sectionPositions.clear()
        var currentSection: String? = null
        for (i in items.indices) {
            val kind = items[i].kind
            if (kind != currentSection) {
                currentSection = kind
                sectionPositions[i] = kind.uppercase()
            }
        }
    }

    private val sectionHeaders: MutableMap<Int, String> = mutableMapOf()

    override fun getCount(): Int {
        return items.size + sectionHeaders.size
    }

    override fun getItem(position: Int): OutputStore.OutputFile? {
        val realIndex = position - countHeadersBefore(position)
        return if (realIndex in items.indices) items[realIndex] else null
    }

    override fun getItemId(position: Int): Long = position.toLong()

    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        // Check if this position is a section header
        val headerText = sectionHeaders[position]
        if (headerText != null) {
            val tv = TextView(context)
            textViewSetup(textView)
            tv.text = headerText.uppercase()
            return tv
        }

        // It's a regular item
        val realIndex = position - countHeadersBefore(position)
        val item = getItem(position) ?: return View(context)

        val view = inflater.inflate(R.layout.item_output, parent, false)

        val cb: CheckBox = view.findViewById(R.id.cbOutputItem)
        val ivThumb: ImageView = view.findViewById(R.id.ivOutputThumbnail)
        val tvName: TextView = view.findViewById(R.id.tvOutputName)
        val tvInfo: TextView = view.findViewById(R.id.tvOutputInfo)
        val ivDelete: ImageView = view.findViewById(R.id.ivOutputDelete)

        cb.isChecked = selectedIndices.contains(position)
        tvName.text = item.name

        val sizeStr = when {
            item.size < 1024 -> "${item.size} B"
            item.size < 1024 * 1024 -> "${item.size / 1024} KB"
            else -> "${item.size / (1024 * 1024)} MB"
        }
        tvInfo.text = "${item.kind.uppercase()} • $size"

        // Set thumbnail for images
        when (item.kind) {
            "image" -> {
                try {
                    val bitmap = BitmapFactory.decodeFile(item.path)
                    if (bitmap != null) {
                        ivThumb.setImageBitmap(bitmap)
                    } else {
                        ivThumb.setImageResource(android.R.drawable.ic_menu_gallery)
                    }
                } catch (e: Exception) {
                    ivThumb.setImageResource(android.R.drawable.ic_menu_gallery)
                }
            }
            "video" -> ivThumb.setImageResource(android.R.drawable.ic_media_video)
            "audio" -> ivThumb.setImageResource(android.R.drawable.ic_media_play)
            "text" -> ivThumb.setImageResource(android.R.drawable.ic_menu_edit)
            else -> ivThumb.setImageResource(android.R.drawable.ic_menu_info_details)
        }

        // Checkbox click - toggle selection
        cb.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                selectedIndices.add(position)
            } else {
                selectedIndices.remove(position)
            }
        }

        // Item click - toggle selection AND open file
        view.setOnClickListener {
            cb.isChecked = !cb.isChecked
            if (cb.isChecked) {
                selectedIndices.add(position)
            } else {
                selectedIndices.remove(position)
            }
            onItemClick(item)
        }

        // Delete button click
        ivDelete.setOnClickListener {
            onDelete(item)
        }

        return view
    }

    private fun countHeadersBefore(position: Int): Int {
        var count = 0
        for (pos in sectionHeaders.keys) {
            if (pos < position) count++ else break
        }
        return count
    }

    /** Selected positions */
    val selectedIndices: MutableSet<Int> = mutableSetOf()

    /** Get selected items */
    fun getSelectedItems(): List<OutputStore.OutputFile> {
        return selectedIndices.sorted().mapNotNull { getItem(it) }
    }

    fun selectAll() {
        selectedIndices = (0 until items.size).toMutableSet()
        notifyDataSetChanged()
    }

    fun clearSelection() {
        selectedIndices.clear()
        notifyDataSetChanged()
    }

    fun deleteSelected() {
        val toDelete = getSelectedItems()
        for (item in toDelete) {
            onDelete(item)
        }
        clearSelection()
        notifyDataSetChanged()
    }

    fun deleteItemAt(position: Int) {
        val realIndex = position - countHeadersBefore(position)
        val item = getItem(position)
        if (item != null) {
            onDelete(item)
        }
    }

    fun removeItem(item: OutputStore.OutputFile) {
        val index = items.indexOf(item)
        if (index >= 0) {
            items.removeAt(index)
            selectedIndices.remove(index)
            rebuildSections()
            notifyDataSetChanged()
        }
    }
}
