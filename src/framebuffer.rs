use std::ops::Deref;

use janus::{GpuResource, texture::TextureView};

pub const MAX_ATTACHMENTS: usize = 8;

pub fn bind_default() {
    bind(FramebufferId::default());
}

pub fn bind(framebuffer: impl Into<FramebufferId>) {
    janus::debug_assert_gl!();
    let framebuffer = framebuffer.into();
    unsafe {
        janus::gl::BindFramebuffer(janus::gl::FRAMEBUFFER, framebuffer.0);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FramebufferId(u32);
impl FramebufferId {
    pub fn bind(self) {
        crate::framebuffer::bind(self);
    }
}
impl From<FramebufferView> for FramebufferId {
    fn from(value: FramebufferView) -> Self {
        value.id
    }
}
impl From<&FramebufferView> for FramebufferId {
    fn from(value: &FramebufferView) -> Self {
        value.id
    }
}
impl From<&Framebuffer> for FramebufferId {
    fn from(value: &Framebuffer) -> Self {
        value.id
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttachmentTexture(TextureView);
impl Default for AttachmentTexture {
    fn default() -> Self {
        Self(TextureView::null(janus::texture::TextureKind::Dim2D))
    }
}
impl From<TextureView> for AttachmentTexture {
    fn from(value: TextureView) -> Self {
        Self(value)
    }
}
impl From<AttachmentTexture> for TextureView {
    fn from(value: AttachmentTexture) -> Self {
        value.0
    }
}
impl Deref for AttachmentTexture {
    type Target = TextureView;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

type ColorAttachments = tinyvec::ArrayVec<[AttachmentTexture; MAX_ATTACHMENTS]>;
type DepthAttachment = Option<AttachmentTexture>;

pub trait HasFramebuffer {
    fn id(&self) -> FramebufferId;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn color_attachments(&self) -> &ColorAttachments;
    fn depth_attachment(&self) -> &DepthAttachment;

    fn bind(&self) {
        crate::framebuffer::bind(self.id());
    }

    fn unbind() {
        crate::framebuffer::bind_default();
    }

    fn color_attachments_len(&self) -> usize {
        self.color_attachments().len()
    }

    fn color_attachment(&self, index: usize) -> AttachmentTexture {
        self.color_attachments()[index]
    }

    fn has_depth(&self) -> bool {
        self.depth_attachment().is_some()
    }

    /// Sets the default read/write buffers state for this Framebuffer.
    ///
    /// * If the Framebuffer has no color attachments:
    ///     The Framebuffer is interpreted as being for a depth-only rendering
    ///     pass, so `GL_NONE` is bound to draw and read buffers at target 0.
    /// * Otherwise:
    ///     Each attachment is mapped linearly to writing buffers (i.e.,
    ///     attachment at index `N` is mapped to target `N` and so on), while
    ///     attachment 0 is bound to the read buffer target.
    ///
    /// Note that OpenGL's read buffer only has a single target.
    fn set_default_buffers_state(&self) {
        if self.color_attachments().is_empty() {
            // this is a depth-only pass
            self.set_write_buffer(None);
            self.set_read_buffer(None);
        } else {
            // map color attachments to draw buffers linearly
            let draw_buffers = std::array::from_fn::<_, MAX_ATTACHMENTS, _>(|i| Some(i as u32));
            self.set_write_buffers(&draw_buffers);
            self.set_read_buffer(Some(0));
        }
    }

    /// Sets the read attachment for blit operations.
    ///
    /// Passing `None` will set the target to `GL_NONE`.
    ///
    /// Note that OpenGL's read buffer only has a single target.
    fn set_read_buffer(&self, attachment_index: Option<u32>) {
        janus::debug_assert_gl!();

        let index = attachment_index.map_or(janus::gl::NONE, |i| janus::gl::COLOR_ATTACHMENT0 + i);
        unsafe {
            janus::gl::NamedFramebufferReadBuffer(self.id().0, index);
        }
    }

    /// Sets the write attachment at shader target 0.
    ///
    /// Passing `None` will set the target to `GL_NONE`.
    fn set_write_buffer(&self, attachment_index: Option<u32>) {
        janus::debug_assert_gl!();

        #[cfg(debug_assertions)]
        if let Some(index) = attachment_index {
            debug_assert!((index as usize) < self.color_attachments_len());
        }

        let index = attachment_index.map_or(janus::gl::NONE, |i| janus::gl::COLOR_ATTACHMENT0 + i);
        unsafe {
            janus::gl::NamedFramebufferDrawBuffer(
                self.id().0,
                janus::gl::COLOR_ATTACHMENT0 + index,
            );
        }
    }

    /// Sets the write attachments for an arbitrary number of shader targets.
    ///
    /// The index of each element in the given `attachment_indices` array
    /// equals to the shader target index. The value of each entry is instead
    /// the index of the attachment of this Framebuffer to be attached to that
    /// target.
    ///
    /// For each `None` element, the target at the index will be set to
    /// `GL_NONE`.
    fn set_write_buffers(&self, attachment_indices: &[Option<u32>]) {
        janus::debug_assert_gl!();

        #[cfg(debug_assertions)]
        for &i in attachment_indices {
            if let Some(i) = i {
                debug_assert!((i as usize) < self.color_attachments_len());
            }
        }

        let attachment_indices = attachment_indices
            .iter()
            .map(|index| {
                if let Some(index) = index {
                    janus::gl::COLOR_ATTACHMENT0 + index
                } else {
                    janus::gl::NONE
                }
            })
            .collect::<tinyvec::ArrayVec<[u32; MAX_ATTACHMENTS]>>();

        let ptr = attachment_indices.as_ptr();
        let len = attachment_indices.len();
        unsafe {
            janus::gl::NamedFramebufferDrawBuffers(self.id().0, len as i32, ptr);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FramebufferView {
    pub id: FramebufferId,
    pub width: u32,
    pub height: u32,
    color_attachments: ColorAttachments,
    depth_attachment: DepthAttachment,
}
impl From<&Framebuffer> for FramebufferView {
    fn from(value: &Framebuffer) -> Self {
        Self::from_framebuffer(value)
    }
}
impl FramebufferView {
    pub const fn from_framebuffer(framebuffer: &Framebuffer) -> Self {
        Self {
            id: framebuffer.id,
            width: framebuffer.width,
            height: framebuffer.height,
            color_attachments: framebuffer.color_attachments,
            depth_attachment: framebuffer.depth_attachment,
        }
    }
}
impl HasFramebuffer for FramebufferView {
    fn id(&self) -> FramebufferId {
        self.id
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn color_attachments(&self) -> &ColorAttachments {
        &self.color_attachments
    }

    fn depth_attachment(&self) -> &DepthAttachment {
        &self.depth_attachment
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FramebufferError {
    #[error("default framebuffer was specified but it is not yet defined")]
    Undefined,
    #[error("an attachment point of the framebuffer is incomplete")]
    IncompleteAttachment,
    #[error("framebuffer has no images attached to it")]
    IncompleteMissingAttachment,
    #[error("draw buffer color attachment is not a valid object type")]
    IncompleteDrawBuffer,
    #[error("read buffer color attachment is not a valid object type")]
    IncompleteReadBuffer,
    #[error("an attachment image produced an internal format error")]
    Unsupported,
    #[error("the number of samples is not equal for all attachments")]
    IncompleteMultisample,
    #[error(
        "framebuffer has a layered attachment but other attachments do not respect the necessary conditions"
    )]
    IncompleteLayerTargets,
}

pub type FramebufferResult = Result<Framebuffer, FramebufferError>;

#[derive(Debug)]
pub struct Framebuffer {
    pub id: FramebufferId,
    pub width: u32,
    pub height: u32,
    color_attachments: ColorAttachments,
    depth_attachment: DepthAttachment,
}
impl Framebuffer {
    pub fn new(
        width: u32,
        height: u32,
        color_attachments: &[TextureView],
        depth_attachment: Option<TextureView>,
    ) -> FramebufferResult {
        janus::debug_assert_gl!();

        let mut id = 0;
        unsafe {
            janus::gl::CreateFramebuffers(1, &mut id);
        };

        for (i, tex) in color_attachments.iter().enumerate() {
            unsafe {
                janus::gl::NamedFramebufferTexture(
                    id,
                    janus::gl::COLOR_ATTACHMENT0 + i as u32,
                    tex.resource_id(),
                    0,
                );
            }
        }
        if let Some(depth_texture) = &depth_attachment {
            unsafe {
                janus::gl::NamedFramebufferTexture(
                    id,
                    janus::gl::DEPTH_ATTACHMENT,
                    depth_texture.resource_id(),
                    0,
                );
            }
        }

        let status = unsafe { janus::gl::CheckNamedFramebufferStatus(id, janus::gl::FRAMEBUFFER) };
        if status != janus::gl::FRAMEBUFFER_COMPLETE {
            use FramebufferError::*;

            return match status {
                janus::gl::FRAMEBUFFER_UNDEFINED => Err(Undefined),
                janus::gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => Err(IncompleteAttachment),
                janus::gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => {
                    Err(IncompleteMissingAttachment)
                }
                janus::gl::FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER => Err(IncompleteDrawBuffer),
                janus::gl::FRAMEBUFFER_INCOMPLETE_READ_BUFFER => Err(IncompleteReadBuffer),
                janus::gl::FRAMEBUFFER_UNSUPPORTED => Err(Unsupported),
                janus::gl::FRAMEBUFFER_INCOMPLETE_MULTISAMPLE => Err(IncompleteMultisample),
                janus::gl::FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS => Err(IncompleteLayerTargets),
                0 => {
                    let mut last_error = 0;
                    #[allow(irrefutable_let_patterns, reason = "glGetError eventually returns 0")]
                    while let err = unsafe { janus::gl::GetError() } {
                        if err != 0 {
                            last_error = err;
                        } else {
                            break;
                        }
                    }
                    match last_error {
                        janus::gl::INVALID_ENUM => {
                            panic!("framebuffer creation unrecoverable error: invalid target")
                        }
                        janus::gl::INVALID_OPERATION => panic!(
                            "framebuffer creation error: invalid framebuffer name is not 0 or an existing framebuffer"
                        ),
                        _ => unreachable!(),
                    };
                }
                _ => unreachable!(),
            };
        }
        // status all ok !!

        let color_attachments = color_attachments
            .iter()
            .copied()
            .filter(|tv| !tv.is_null())
            .map(From::from)
            .collect();
        let depth_attachment = depth_attachment.map(From::from);

        Ok(Self {
            id: FramebufferId(id),
            width,
            height,
            color_attachments,
            depth_attachment,
        })
    }

    pub const fn as_view(&self) -> FramebufferView {
        FramebufferView::from_framebuffer(self)
    }
}
impl HasFramebuffer for Framebuffer {
    fn id(&self) -> FramebufferId {
        self.id
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn color_attachments(&self) -> &ColorAttachments {
        &self.color_attachments
    }

    fn depth_attachment(&self) -> &DepthAttachment {
        &self.depth_attachment
    }
}
impl Drop for Framebuffer {
    fn drop(&mut self) {
        janus::debug_assert_gl!();
        unsafe {
            janus::gl::DeleteFramebuffers(1, &self.id.0);
        }
    }
}
