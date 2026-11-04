package eu.odysseus.diffusion

import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Typeface
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.TextView

/**
 * Adapter for the Outputs dialog list.
 *
 * Items are grouped by kind with a section header in front of each group, so
 * the adapter's [getCount] is larger than the item count. Selection is keyed
 * by *display* position; [getItem] maps a display position back to its item.
 */
class OutputsAdapter(
    private val context: Context,
    private val items: MutableList<OutputFile>,
    private val onItemClick: (OutputFile) -> Unit,
    private val onDelete: (OutputFile) -> Unit,
) : ArrayAdapter<OutputFile>(context, 0, items) {

    private val inflater: LayoutInflater = LayoutInflater.from(context)

    /** Section header label by display position. Rebuilt by [rebuildSections]. */
    private val headerAt: MutableMap<Int, String> = mutableMapOf()

    /** Selected display positions. */
    val selectedIndices: MutableSet<Int> = mutableSetOf()

    init {
        rebuildSections()
    }

    /** Recompute header positions after the item list changed. */
    fun rebuildSections() {
        headerAt.clear()
        var displayPos = 0
        var currentSection: String? = null
        for (item in items) {
            if (item.kind != currentSection) {
                currentSection = item.kind
                headerAt[displayPos] = item.kind
                displayPos++
            }
            displayPos++
        }
    }

    override fun getCount(): Int = items.size + headerAt.size

    private fun headersBefore(position: Int): Int = headerAt.keys.count { it < position }

    override fun getItem(position: Int): OutputFile? {
        val realIndex = position - headersBefore(position)
        return if (realIndex in items.indices) items[realIndex] else null
    }

    override fun getItemId(position: Int): Long = position.toLong()

    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        val headerText = headerAt[position]
        if (headerText != null) {
            val tv = TextView(context)
            tv.text = headerText.uppercase()
            tv.setTextColor(0xFF57606A.toInt())
            tv.textSize = 12f
            tv.setTypeface(null, Typeface.BOLD)
            tv.setPadding(16, 24, 16, 8)
            return tv
        }

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
        tvInfo.text = "${item.kind.uppercase()} \u2022 $sizeStr"

        // Thumbnail for images. item.uri is a file:// URI built by OutputStore.list().
        when (item.kind) {
            "image" -> {
                val bitmap = try {
                    BitmapFactory.decodeFile(item.uri.path)
                } catch (e: Exception) {
                    null
                }
                if (bitmap != null) {
                    ivThumb.setImageBitmap(bitmap)
                } else {
                    ivThumb.setImageResource(android.R.drawable.ic_menu_gallery)
                }
            }
            "video" -> ivThumb.setImageResource(android.R.drawable.ic_media_ff)
            "audio" -> ivThumb.setImageResource(android.R.drawable.ic_media_play)
            "text" -> ivThumb.setImageResource(android.R.drawable.ic_menu_edit)
            else -> ivThumb.setImageResource(android.R.drawable.ic_menu_info_details)
        }

        cb.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) selectedIndices.add(position) else selectedIndices.remove(position)
        }

        // Item click: toggle selection and open the file.
        view.setOnClickListener {
            cb.isChecked = !cb.isChecked
            onItemClick(item)
        }

        ivDelete.setOnClickListener { onDelete(item) }

        return view
    }

    /** Replace the whole list (sort spinner, delete-all). */
    fun setItems(newItems: List<OutputFile>) {
        items.clear()
        items.addAll(newItems)
        selectedIndices.clear()
        rebuildSections()
        notifyDataSetChanged()
    }

    /** Currently selected items. */
    fun getSelectedItems(): List<OutputFile> =
        selectedIndices.sorted().mapNotNull { getItem(it) }

    fun selectAll() {
        selectedIndices.clear()
        for (p in 0 until count) {
            if (headerAt[p] == null) selectedIndices.add(p)
        }
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

    fun removeItem(item: OutputFile) {
        val index = items.indexOf(item)
        if (index >= 0) {
            items.removeAt(index)
            selectedIndices.clear()
            rebuildSections()
            notifyDataSetChanged()
        }
    }
}
